from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import os
import base64
import pandas as pd
import numpy as np
import tenseal as ts
import requests
import time
from django.core.files.storage import default_storage
from scipy.stats import chi2 as chi2_distribution
import json

# Helper functions
def read_genotypes(file_path):
    with open(file_path, "r") as f:
        return f.read().split()  # Assuming space-separated genotypes

def encrypt_in_chunks(data, chunk_size, context):
    encrypted_chunks = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk += [-1] * (chunk_size - len(chunk))
        encrypted_chunks.append(ts.ckks_vector(context, chunk))
    return encrypted_chunks

def write_data(file_name: str, data: bytes):
    data = base64.b64encode(data)
    with open(file_name, "wb") as f: 
        f.write(data)
  
def read_data(file_name: str) -> bytes:
    with open(file_name, "rb") as f:
        data = f.read()
    return base64.b64decode(data)

def compute_allele_counts(aa, ag, gg):
    return {
        "A": 2 * aa + ag,
        "G": 2 * gg + ag
    }

def hwe_chi2(aa, ag, gg):
    n = aa + ag + gg
    if n == 0:
        return 0
    p = (2 * aa + ag) / (2 * n)
    q = 1 - p
    exp_aa = p**2 * n
    exp_ag = 2 * p * q * n
    exp_gg = q**2 * n
    return ((aa - exp_aa) ** 2 / exp_aa +
            (ag - exp_ag) ** 2 / exp_ag +
            (gg - exp_gg) ** 2 / exp_gg)

@csrf_exempt
def upload_file(request):
    if request.method == 'POST':
        enc_case = request.POST.getlist('enc_case')
        enc_control = request.POST.getlist('enc_control')

        if not enc_case or not enc_control:
            return JsonResponse({'error': 'Encrypted data is missing'}, status=400)

        upload_dir = os.path.join(settings.BASE_DIR, 'uploaded_files')
        os.makedirs(upload_dir, exist_ok=True)  # Ensure the directory exists

        # Save encrypted case data
        enc_case_path = os.path.join(upload_dir, 'encrypted_case.txt')
        with open(enc_case_path, 'w') as enc_case_file:
            for chunk in enc_case:
                enc_case_file.write(chunk + '\n')

        # Save encrypted control data
        enc_control_path = os.path.join(upload_dir, 'encrypted_control.txt')
        with open(enc_control_path, 'w') as enc_control_file:
            for chunk in enc_control:
                enc_control_file.write(chunk + '\n')

        return JsonResponse({
            'message': 'Encrypted files saved successfully!',
            'enc_case_path': enc_case_path,
            'enc_control_path': enc_control_path
        })

    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def upload_and_encrypt(request):
    if request.method == 'POST':
        case_file = request.FILES.get('case_file')
        controls_file = request.FILES.get('controls_file')
        algorithm = request.POST.get('algorithm')

        if not case_file or not controls_file or not algorithm:
            return JsonResponse({'error': 'Missing required files or algorithm'}, status=400)

        try:
            # Save files and get paths
            case_file_path = default_storage.save(os.path.join(settings.MEDIA_ROOT, case_file.name), case_file)
            controls_file_path = default_storage.save(os.path.join(settings.MEDIA_ROOT, controls_file.name), controls_file)

            # Read and process genotypes
            case_genotypes = read_genotypes(case_file_path)
            control_genotypes = read_genotypes(controls_file_path)

            genotype_mapping = {"AA": 0, "AG": 1, "GG": 2}
            case_numeric = [genotype_mapping.get(g, -1) for g in case_genotypes]
            control_numeric = [genotype_mapping.get(g, -1) for g in control_genotypes]

            # Create encryption context
            context = ts.context(
                scheme=ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=32768,
                coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60] if algorithm != 'algorithm5' else
                                    [60] + [40] * 16 + [60]
            )
            if algorithm == 'algorithm1':
                context.global_scale = 2**30
            else:
                context.global_scale = 2**40
            context.generate_galois_keys()

            # Store contexts
            secret_context = context.serialize(save_secret_key=True)
            write_data("secret.txt", secret_context)
            
            context.make_context_public()
            public_context = context.serialize()
            write_data("public.txt", public_context)

            # Load public context for encryption
            context = ts.context_from(read_data("public.txt"))

            # Start encryption timing
            encryption_start = time.time()
            
            # Encrypt data in chunks
            chunk_size = 8192
            enc_case_chunks = encrypt_in_chunks(case_numeric, chunk_size, context)
            enc_control_chunks = encrypt_in_chunks(control_numeric, chunk_size, context)

            # End encryption timing
            encryption_end = time.time()
            print(f"\nEncryption time: {encryption_end - encryption_start:.2f} seconds")

            # Serialize encrypted data
            enc_case_serialized = [base64.b64encode(chunk.serialize()).decode('utf-8') for chunk in enc_case_chunks]
            enc_control_serialized = [base64.b64encode(chunk.serialize()).decode('utf-8') for chunk in enc_control_chunks]
            context_serialized = base64.b64encode(public_context).decode('utf-8')

           
            # Define file paths
            upload_dir = os.path.join(settings.BASE_DIR, 'uploaded_files')
            os.makedirs(upload_dir, exist_ok=True)
            enc_case_path = os.path.join(upload_dir, 'enc_case.txt')
            enc_control_path = os.path.join(upload_dir, 'enc_control.txt')
            context_path = os.path.join(upload_dir, 'public_context.txt')

            with open(enc_case_path, 'w') as f:
                json.dump(enc_case_serialized, f)
            with open(enc_control_path, 'w') as f:
                json.dump(enc_control_serialized, f)
            with open(context_path, 'w') as f:
                f.write(context_serialized)

            # Delete original plaintext files after encryption
            try:
                os.remove(case_file_path)
                os.remove(controls_file_path)
                print(f"Deleted plaintext files: {case_file_path}, {controls_file_path}")
            except Exception as e:
                print(f"Error deleting plaintext files: {e}")

            # Send to Flask server
            flask_url = 'http://127.0.0.1:5000/upload/'
            with open(enc_case_path, 'rb') as case_f, open(enc_control_path, 'rb') as control_f, open(context_path, 'rb') as ctx_f:
                files = {
                    'caseFile': ('enc_case.txt', case_f, 'text/plain'),
                    'controlsFile': ('enc_control.txt', control_f, 'text/plain'),
                    'contextFile': ('public_context.txt', ctx_f, 'text/plain')
                }
                data = {'algorithm': algorithm}
                try:
                    flask_response = requests.post(flask_url, files=files, data=data)
                    flask_result = flask_response.json() if flask_response.ok else flask_response.text
                except Exception as e:
                    flask_result = {'error': str(e)}
            

            print("Computation time from server:", flask_result.get("computation_time"))
            # Debugging: Log data sent to the server
            print("Data sent to Flask server:")
            print("Case file path:", enc_case_path)
            print("Control file path:", enc_control_path)
            print("Context file path:", context_path)
            print("Algorithm:", algorithm)

            # Debugging: Log server response
            # print("Response received from Flask server:", flask_result)
            required_keys = [
                "message", "algorithm", "results"
            ]
            for key in required_keys:
                if key not in flask_result:
                    return JsonResponse({
                        'error': f'Missing key in Flask response: {key}',
                        'flask_result': flask_result
                    }, status=500)
                    
            # Check for required keys in results
            result_keys = [
                "case_aa", "case_ag", "case_gg",
                "control_aa", "control_ag", "control_gg",
                "allelic_contingency_table"
            ]
            for key in result_keys:
                if key not in flask_result["results"]:
                    return JsonResponse({
                        'error': f'Missing key in Flask results: {key}',
                        'flask_result': flask_result
                    }, status=500)

            # Restore private key context for decryption
            decryption_context = ts.context_from(read_data("secret.txt"))
            
            # Start decryption timing
            decryption_start = time.time()
            
            # Helper function to deserialize and decrypt
            def decrypt_count(serialized_count):
                enc_vector = ts.ckks_vector_from(decryption_context, base64.b64decode(serialized_count))
                decrypted = enc_vector.decrypt()
                return int(round(decrypted[0])) if len(decrypted) > 0 else 0

            # Decrypt counts from Flask response
            decrypted_counts = {
                "case": {
                    "AA": decrypt_count(flask_result["results"]["case_aa"]),
                    "AG": decrypt_count(flask_result["results"]["case_ag"]),
                    "GG": decrypt_count(flask_result["results"]["case_gg"])
                },
                "control": {
                    "AA": decrypt_count(flask_result["results"]["control_aa"]),
                    "AG": decrypt_count(flask_result["results"]["control_ag"]),
                    "GG": decrypt_count(flask_result["results"]["control_gg"])
                }
            }

            # End decryption timing
            decryption_end = time.time()
            print(f"Decryption time: {decryption_end - decryption_start:.2f} seconds\n")

            # Compute allele counts for cases and controls
            case_alleles = compute_allele_counts(
                decrypted_counts["case"]["AA"],
                decrypted_counts["case"]["AG"],
                decrypted_counts["case"]["GG"]
            )

            control_alleles = compute_allele_counts(
                decrypted_counts["control"]["AA"],
                decrypted_counts["control"]["AG"],
                decrypted_counts["control"]["GG"]
            )

            # Create contingency table
            contingency_table = pd.DataFrame({
                "Cases": [case_alleles["A"], case_alleles["G"]],
                "Controls": [control_alleles["A"], control_alleles["G"]]
            }, index=["A", "G"])

            # Print contingency table and genotype counts for debugging
            print("\nAllelic Contingency Table:")
            print(contingency_table)
            print("\nGenotype Counts:")
            print("Cases:", decrypted_counts["case"])
            print("Controls:", decrypted_counts["control"])

            # Calculate statistics
            # Allelic Odds Ratio (AOR)
            aor = (case_alleles["A"] * control_alleles["G"]) / (control_alleles["A"] * case_alleles["G"]) if control_alleles["A"] * case_alleles["G"] != 0 else float('inf')

            # Chi-square test
            chi2_table = contingency_table.values.astype(np.float64)
            n = np.sum(chi2_table)
            # Calculate expected frequencies
            row_sums = np.sum(chi2_table, axis=1)
            col_sums = np.sum(chi2_table, axis=0)
            expected = np.outer(row_sums, col_sums) / n
            # Calculate chi-square statistic
            chi2 = np.sum(np.where(expected > 0,
                                 (chi2_table - expected) ** 2 / expected,
                                 0.0))

            # Minor Allele Frequencies
            maf_case = min(case_alleles["A"], case_alleles["G"]) / (case_alleles["A"] + case_alleles["G"])
            maf_control = min(control_alleles["A"], control_alleles["G"]) / (control_alleles["A"] + control_alleles["G"])

            # HWE Chi-square test
            hwe_case = hwe_chi2(
                decrypted_counts["case"]["AA"],
                decrypted_counts["case"]["AG"],
                decrypted_counts["case"]["GG"]
            )
            hwe_control = hwe_chi2(
                decrypted_counts["control"]["AA"],
                decrypted_counts["control"]["AG"],
                decrypted_counts["control"]["GG"]
            )

            # Print final statistics
            print("\nStatistical Results:")
            print(f"Allelic Odds Ratio: {aor:.4f}")
            print(f"Chi-square statistic: {chi2:.4f}")
            print(f"MAF Case: {maf_case:.4f}")
            print(f"MAF Control: {maf_control:.4f}")
            print(f"HWE Chi-square Case: {hwe_case:.4f}")
            print(f"HWE Chi-square Control: {hwe_control:.4f}")

            # Prepare the results data
            try:
                # Convert contingency table to a simpler format
                contingency_data = {
                    'alleles': ['A', 'G'],
                    'cases': contingency_table['Cases'].tolist(),
                    'controls': contingency_table['Controls'].tolist()
                }
                
                results_data = {
                    'genotype_counts': decrypted_counts,
                    'allele_counts': {
                        'case': case_alleles,
                        'control': control_alleles
                    },
                    'contingency_table': contingency_data,
                    'statistics': {
                        'allelic_odds_ratio': float(aor),
                        'chi_square_statistic': float(chi2),
                        'minor_allele_frequency_case': float(maf_case),
                        'minor_allele_frequency_control': float(maf_control),
                        'hardy-Weinberg_equilibrium_case': float(hwe_case),
                        'hardy-Weinberg_equilibrium_control': float(hwe_control)
                    }
                }
                print("Debug - Results data:", results_data)  # Debug print
                
                # Store results in session
                request.session['results_data'] = results_data
                return JsonResponse({
                    'success': True,
                    'redirect_url': '/results/'
                })

            except Exception as render_error:
                print(f"Error preparing template data: {str(render_error)}")
                return JsonResponse({
                    'error': f'Failed to render template: {str(render_error)}'
                }, status=500)

        except Exception as e:
            return JsonResponse({
                'error': f'Failed to process request: {str(e)}',
                'traceback': str(e.__traceback__)
            }, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)


def results_view(request):
    results_data = request.session.get('results_data')
    
    # Add statistics descriptions
    stat_descriptions = {
        'allelic_odds_ratio': 'Odds of an allele appearing in cases versus controls.',
        'chi_square_statistic': 'Chi-square test result measuring association strength.',
        'minor_allele_frequency_case': 'Minor Allele Frequency among cases.',
        'minor_allele_frequency_control': 'Minor Allele Frequency among controls.',
        'hardy-Weinberg_equilibrium_case': 'Hardy-Weinberg Equilibrium chi-square in cases.',
        'hardy-Weinberg_equilibrium_control': 'Hardy-Weinberg Equilibrium chi-square in controls.'
    }

    context = {
        'error': None if results_data else 'No results data available',
        'stat_descriptions': stat_descriptions
    }

    if results_data:
        context.update(results_data)

    return render(request, 'results.html', context)