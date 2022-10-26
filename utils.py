from os import makedirs


def save_params_file(path, *paramdicts):
    text = ""
    makedirs(path, exist_ok=True)
    with open(f"{path}/config.txt", "a") as f:
        for d in paramdicts:
            for param in d:
                f.write(f"{param} = {d[param]} \n")
                text+=f"{param} = {d[param]} \n"
    return text

