


import os

def cleanup(log_file):
    os.popen('git pull')
    lines = open(log_file, 'r').read().split('\n')              
    cleaned_lines = [l for l in lines[:-1] if l[0] not in ['<','>', "="]] 
    unique_lines = []


    completed = []

    for line in cleaned_lines:
        vals = line.split(',')
        if vals[1] == 'complete':
            completed.append(vals[0])


    for line in cleaned_lines:
        if line not in unique_lines:
            vals = line.split(',')
            if vals[1] == 'not run' and vals[0] in completed:
                continue

            if vals[1][:11] == 'in progress' and vals[0] in completed:
                continue

            unique_lines.append(line)

    with open(log_file,'w') as fd:              
        fd.write('\n'.join(unique_lines))

    os.popen('git commit -am "routine cleanup of tracking file"')
    os.popen('git pull')
    os.popen('git push')



    return unique_lines



def cleanup_no_git(log_file):
    # os.popen('git pull')
    lines = open(log_file, 'r').read().split('\n')              
    cleaned_lines = [l for l in lines[:-1] if l[0] not in ['<','>', "="]] 
    unique_lines = []


    completed = []

    for line in cleaned_lines:
        vals = line.split(',')
        if vals[1] == 'complete':
            completed.append(vals[0])


    for line in cleaned_lines:
        if line not in unique_lines:
            vals = line.split(',')
            if vals[1] == 'not run' and vals[0] in completed:
                continue

            if vals[1][:11] == 'in progress' and vals[0] in completed:
                continue

            unique_lines.append(line)

    with open(log_file,'w') as fd:              
        fd.write('\n'.join(unique_lines))

    # os.popen('git commit -am "routine cleanup of tracking file"')
    # os.popen('git pull')
    # os.popen('git push')



    return unique_lines
