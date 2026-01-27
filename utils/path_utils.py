import os, json

def get_chrome_download_path():
    if os.name == 'nt':
        raise OSError('Windows system is not yet supported')
        # path = os.path.join(os.getenv('LOCALAPPDATA'), 'Google', 'Chrome', 'User Data', 'Default', 'Preferences')
    elif os.name == 'posix':
        path = os.path.expanduser('~/.config/google-chrome/Default/Preferences')
        if not os.path.exists(path):
            path = os.path.expanduser('~/Library/Application Support/Google/Chrome/Default/Preferences')
    else:
        raise OSError('Unsupported OS system')

    with open(path, 'r', encoding='utf-8') as f:
        prefs = json.load(f)
        return prefs.get('download', {}).get('default_directory')
