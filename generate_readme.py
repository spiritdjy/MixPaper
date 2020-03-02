# coding: utf-8
"""
自动生成ReadMe.md
"""


def generate():
    with open('README.md') as fi:
        for line in fi:
            line = line.strip()
            if line.startswith('####'):
                print(line)
                generate_paper(line)
            else:
                if line.startswith('-'):
                    continue
                
                print(line)
                

def generate_paper(line):
    content = line.split('(')[1].split(')')[0]
    with open(content) as fi:
        for line in fi:
            if line.startswith('#') and all(i in line for i in '[]()'):
                line = line.strip('# \n').replace('../', '')
                print('- ' + line)


if __name__ == "__main__":
    """
    python -m generate_readme generate
    """
    from fire import Fire
    Fire()
