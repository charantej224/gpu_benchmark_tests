def write_file(filename, content):
    with open(filename, 'a+') as f:
        f.write(content)
        f.close()
