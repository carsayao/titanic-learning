## How to set up and use `virtual env`

Virtualenv creates a python environment in your current directory. To remove libraries from your computer, just delete the env/ dir.

    pip3 install virtualenv

    virtualenv -p python3 env

This will install virtualenv using pip3 and create a an `env` folder that will contain the libraries brought it. `-p` indicates python version, `env` can be any name you want.

    source env/bin/activate

This will put your shell into the virtual environment where you will have the libraries brought it. You may have to use different `activate` depending on your shell. For instance sourcing with `fish` is `. env/bin/activate.fish`.

    pip3 install -r requirements.txt

This will install the libraries that are found in the `requirements.txt`. When you are done working, use `deactivate` to get out of the virtual environment.

