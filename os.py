import os
import sys
import shutil



def rm_models(model):
    path = './models'
    countries = os.listdir(path)
    print(countries)
    for c in countries:
        tmp = path +'/'+ c + '/' + model
        print(tmp)
        #os.rmdir(tmp)
        try:
            os.remove(tmp)
        except:
            shutil.rmtree(tmp)


def main():
    model = sys.argv[1]
    rm_models(model)

if __name__ == "__main__":
    main()   