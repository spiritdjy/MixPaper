# coding: utf-8
"""
自动生成ReadMe.md
"""


def generate():
    c = 0
    with open('README.md') as fi:
        for line in fi:
            line = line.strip()
            if line.startswith('####'):
                print(line)
                c += generate_paper(line)
            else:
                if line.startswith('-'):
                    continue
                
                print(line)
    print("ALL -->", c)
                

def generate_paper(line):
    ret = []
    content = line.split('(')[1].split(')')[0]
    with open(content) as fi:
        for line in fi:
            if line.startswith('#') and all(i in line for i in '[]()'):
                line = line.strip('# \n').replace('../', '')
                file = line.split('(')[1].split(')')[0]
                if '/' in file and 'http' not in file:
                    try:
                        with open(file) as f:
                            lines = f.readlines()
                            c = len([i for i in lines if i.startswith('#')])
                            if c >= 4 and len(lines) > 30:
                                ret.append('- ' + line)
                    except:
                        pass
        
    ret.sort(reverse=True)
    for i in ret:
        print(i)
        
    return len(ret)


if __name__ == "__main__":
    """
    python -m generate_readme generate
    """
    from fire import Fire
    Fire()
