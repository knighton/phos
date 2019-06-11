def parse_kwarg_value(s):
    try:
        return int(s)
    except:
        pass
    try:
        return float(s)
    except:
        pass
    return s


def parse_kwargs(s):
    kwargs = {}
    ss = s.split(',')
    for s in ss:
        index = s.index('=')
        k = s[:index]
        v = parse_kwarg_value(s[index + 1:])
        kwargs[k] = v
    return kwargs


def parse_class_and_kwargs(module, s):
    index = s.find(':')
    if index == -1:
        klass = s
        kwargs = {}
    else:
        klass = s[:index]
        kwargs = parse_kwargs(s[index + 1:])
    klass = getattr(module, klass)
    return klass, kwargs
