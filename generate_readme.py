# coding: utf-8
"""
自动生成ReadMe.md
"""


def generate():
    c = 0
    with open('README.md') as fi:
        for line in fi:
            line = line.strip()
            if line.startswith('# ') or line.startswith('## '):
                print(line)
                print()
                continue
            if line.startswith('### '):
                print(line)
                print()
                c += generate_paper(line)
            else:
                if line.startswith('-'):
                    continue
                if line.startswith('####'):
                    continue
                if not line:
                    continue
                if line:
                    print(line)
    print("ALL -->", c)
                

def generate_paper(line):
    ret = []
    content = line.split('(')[1].split(')')[0]
    with open(content) as fi:
        lines = fi.readlines()
        mark_count = set(i.count('#') for i in lines if i.count('#'))
        for line in lines:
            line = line.strip()
            c = line.count('#')
            if c not in [0, min(mark_count), max(mark_count)]:
                if ret and ret[-1] != '':
                    ret.append('')
                ret.append('#### ' + line.strip().replace('#', '') + '\n')
                continue
            if line.startswith('#') and all(i in line for i in '[]()'):
                line = line.strip('#').strip().replace('../', '')
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
            
    # ret.sort(reverse=True)
    for i in ret:
        print(i)
    if ret and ret[-1] != '':
        print()
        
    return len([i for i in ret if i.startswith('-')])


if __name__ == "__main__":
    """
    python -m generate_readme generate
    """
    from fire import Fire
    Fire()
