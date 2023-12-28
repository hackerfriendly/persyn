def doiify(text):
    ''' Turn DOIs into doi.org links '''
    return re.sub(
        r"(https?://doi\.org/)?(10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+)",
        lambda match: match.group(0) if match.group(2) is None else f'https://doi.org/{match.group(2)}',
        text
    ).rstrip('.')
