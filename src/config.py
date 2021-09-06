from configparser import ConfigParser


def config(ini_file='./database.ini', section='postgresql'):
    parser = ConfigParser()
    parser.read(ini_file)

    if not parser.has_section(section):
        raise Exception(f'Section {section} not found in the {ini_file} file')

    params = parser.items(section)
    return {param[0]: param[1] for param in params}