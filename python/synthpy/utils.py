import numpy as np


def fread(line,fmt):
    """
    Read the values in a string into variables using a format string.
    (1X, A8, 4I5, F9.3, F15.3, 2F9.1)
    """
    # Transform the format string into an array
    if fmt.startswith('('):
        fmt = fmt[1:]
    if fmt.endswith(')'):
        fmt = fmt[:-1]
    fmtarr = fmt.split(',')
    # Expand repeat values, e.g. 2I5 -> I5,I5
    fmtlist = []
    for i in range(len(fmtarr)):
        fmt1 = fmtarr[i].strip()
        # X format starts with a number
        if fmt1.find('X')>-1:
            fmtlist.append(fmt1)
            continue
        # Repeats
        if fmt1[0].isnumeric():
            ind, = np.where(np.char.array(list(fmt1)).isalpha()==True)
            ind = ind[0]
            num = int(fmt1[0:ind])
            
            fmtlist += list(np.repeat(fmt1[ind:],num))
        else:
            fmtlist.append(fmt1)
    # Start the output tuple
    out = ()
    count = 0
    nline = len(line)
    for i in range(len(fmtlist)):
        fmt1 = fmtlist[i]
        out1 = None
        # Ignore X formats
        if fmt1.find('X')==-1:
            if fmt1[0]=='A':
                num = int(fmt1[1:])
                if count+num <= nline:
                    out1 = line[count:count+num]
            elif fmt1[0]=='I':
                num = int(fmt1[1:])
                if count+num <= nline:                
                    out1 = int(line[count:count+num])
            else:
                ind = fmt1.find('.')
                num = int(fmt1[1:ind])
                if count+num <= nline:            
                    out1 = float(line[count:count+num])
            out = out + (out1,)
            count += num
            
        # X formats, increment the counter
        else:
            ind = fmt1.find('X')
            num = int(fmt1[0:ind])
            count += num

    return out

def toroman(number):
    """ Function to convert integer to Roman numeral."""
    # https://www.geeksforgeeks.org/python-program-to-convert-integer-to-roman/
    num = [1, 4, 5, 9, 10, 40, 50, 90,
        100, 400, 500, 900, 1000]
    sym = ["I", "IV", "V", "IX", "X", "XL",
        "L", "XC", "C", "CD", "D", "CM", "M"]
    i = 12
    rnum = ''    
    while number:
        div = number // num[i]
        number %= num[i]
        while div:
            rnum += sym[i]
            div -= 1
        i -= 1
    return rnum

def fromroman(rnum):
    """ Function to convert from Roman numeral to integer."""
    # https://www.geeksforgeeks.org/python-program-for-converting-roman-numerals-to-decimal-lying-between-1-to-3999/
    num = [1, 5, 10, 50, 100, 500, 1000]
    sym = ["I", "V", "X", "L", "C", "D", "M"]
    sym2num = dict(zip(sym,num))    
    number = 0
    i = 0 
    while (i < len(rnum)):
        # Getting value of symbol s[i]
        s1 = sym2num(rnum[i])
        if (i + 1 < len(rnum)):
            # Getting value of symbol s[i + 1]
            s2 = sym2num(rnum[i + 1])
            # Comparing both values
            if (s1 >= s2):
                # Value of current symbol is greater
                # or equal to the next symbol
                number += s1
                i += 1
            else:
                # Value of current symbol is greater
                # or equal to the next symbol
                number += s2 - s1
                i += 2
        else:
            number += s1
            i += 1
    return number
