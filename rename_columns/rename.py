import pandas as pd
import matplotlib.pyplot as plt
import random

def get_value_number(oldname: str) -> str:
    number = oldname.split(".")[-1]
    if number == "inf":
        number = "\\infty"
    return number

def get_stat_name(oldname: str) -> str:
    if "autocorr" in oldname:
        return "auto$-$correlation"
    elif "mean" in oldname:
        return "{mean}"
    elif "std" in oldname:
        return "standard$ $deviation"
    elif "chisq" in oldname:
        return "chi$-$squared"
    elif "moment" in oldname:
        moment_num = get_num_with_ordinal_indicator(oldname)
        return moment_num + "$ ${moment}"
    elif "renyi" in oldname:
        number = get_value_number(oldname)
        return f"R\\'enyi's$ $entropy$ $(\\alpha = {number})"
    elif "tsallis" in oldname:
        number = get_value_number(oldname)
        return f"Tsallis$ $entropy$ $(q = {number})"
    elif "shannon" in oldname or "shanon" in oldname:
        return "Shannon's$ $entropy"
    elif "extropy" in oldname:
        return "extropy"
    elif "skew" in oldname:
        return "skewness"
    elif "kurtosis" in oldname:
        return "kurtosis"
    elif "montecarlo" in oldname:
        return "Monte$-$Carlo$ $\pi"
    else:
        raise Exception(f"Unknown statistic {oldname}")

def get_fourier_stat_name(oldname: str) -> str:
    return f"statistic$ ${get_stat_name(oldname)}"
    
def get_num_with_ordinal_indicator(oldname: str) -> str:
    number = oldname.split(".")[-1]
    if number == "1":
        return "1^{st}"
    elif number == "2":
        return "2^{nd}"
    elif number == "3":
        return "3^{rd}"
    elif number == "inf":
        return "\\inf"
    else:
        return number + "^{th}"

def get_onebyte_or_fourbytes(oldname: str) -> str:
    if "4byte" in oldname:
        return "(4$ $bytes)"
    else:
        return "(1$ $byte)"

def fourier_column(oldname: str) -> str:
    byte_str = get_onebyte_or_fourbytes(oldname)
    if ".value." in oldname:
        value_number = get_value_number(oldname)
        return f"Fourier$ $raw$ $value$ ${value_number}$ ${byte_str}"
    elif ".stat." in oldname:
        stat_name = get_fourier_stat_name(oldname)
        return f"Fourier$ ${stat_name}$ ${byte_str}"
    else:
        raise Exception(f"Unexpected feature name {oldname}")
        

def baseline_column(oldname: str) -> str:
    return get_stat_name(oldname)

def advanced_column(oldname: str) -> str:
    return f"{get_stat_name(oldname)}"

def should_discard(colname: str) -> bool:
    colname = colname.lower()
    for i in ["head", "tail", "begin", "end", "filesize"]:
        if i in colname:
            return True
    return False

def rename_column(oldname: str) -> str:
    if should_discard(oldname):
        return None
    oldname = oldname.lower()
    if oldname.startswith("fourier"):
        return f"${fourier_column(oldname)}$"
    elif oldname.startswith("baseline"):
        return f"${baseline_column(oldname)}$"
    elif oldname.startswith("advanced"):
        return f"${advanced_column(oldname)}$"
    else:
        print("Unknown column", oldname)
        return None

if __name__ == "__main__":
    df = pd.read_csv("s9.n1.ransomware.SODINOKIBI.csv.gz")

    newcols = [c for c in df.columns if not should_discard(c)]
    newcols = [rename_column(s) for s in newcols if rename_column(s) is not None]
    newcols = list(set(newcols))
    values = [random.randint(0, 10) for i in range(len(newcols))]

    df = pd.DataFrame({"F": newcols, "V": values})
    df.plot.barh(x="F", y="V")
    plt.tight_layout()
    plt.show()