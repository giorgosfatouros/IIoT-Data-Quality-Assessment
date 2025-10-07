version_stop_words = ['%MAJOR_VERSION%', '%MINOR_VERSION%']

def updateMinVersion(version, src=''):
    # Remove possible SNAPSHOT
    version = version.split("-")[0] if "-" in version else version
    print(version)
    versions = version.split(".")
    majorVersion = versions[0]
    minorVersion = versions[1]
    print("Full Version: {} Major: {} Minor: {}".format(version, majorVersion, minorVersion))
    versionsFileTemplate = '{}templates/versions.py.template'.format(src)
    versionsFile = '{}pyLeanxcale/versions.py'.format(src)
    with open(versionsFileTemplate) as template:
        templateLines = template.readlines()
    with open(versionsFile, 'w') as toWrite:
        [toWrite.write(line.replace(stop_word, majorVersion if stop_word is version_stop_words[0] else minorVersion)) if
         stop_word in line else line for stop_word in version_stop_words for line in templateLines]
