from django import template

register = template.Library()

@register.filter
def get_item(lst, i):
    try:
        return lst[i]
    except:
        return None

@register.filter
def replace(value, args):
    """
    Replace one string with another in a given value
    Usage: {{ value|replace:"_" " " }}
    """
    if len(args.split()) != 2:
        return value
    old, new = args.split()
    return str(value).replace(old, new)

@register.filter
def replace(value, arg):
    """
    Replaces all instances of arg with spaces and capitalizes each word
    """
    return value.replace(arg, ' ').title()

@register.filter
def get(dict_obj, key):
    return dict_obj.get(key)