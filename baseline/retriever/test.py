import click

@click.command()
@click.option("--a", default="a", help="a", required=False)
@click.option("--b", default="b", help="b", required=False)
@click.option("--c", default="c", help="c", required=False)
def main(a,b,c):
    print("a",a)
    print("b",b)
    print("c",c)

if __name__=="__main__":
    main(obj=None)