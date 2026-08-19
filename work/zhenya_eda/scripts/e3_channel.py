"""E3. Согласованность флагов каналов и семантика «пустой» строки."""
import polars as pl
df = pl.read_parquet("train.parquet")
n = df.height

def cnt(expr, label):
    k = df.filter(expr).height
    print(f"  {label:52s} {k:>10,}  ({100*k/n:.3f}%)")

print("=== search-флаг против содержимого ===")
cnt((pl.col("search") == 1) != (pl.col("searches") > 0), "search != (searches>0)")
cnt((pl.col("gmv_search") > 0) & (pl.col("search") == 0), "gmv_search>0 ПРИ search=0")
cnt((pl.col("search_to_cart") > 0) & (pl.col("search") == 0), "search_to_cart>0 ПРИ search=0")
cnt((pl.col("search_to_ord") > 0) & (pl.col("search") == 0), "search_to_ord>0 ПРИ search=0")

print("\n=== cat-флаг против содержимого ===")
cnt((pl.col("gmv_cat") > 0) & (pl.col("cat") == 0), "gmv_cat>0 ПРИ cat=0")
cnt((pl.col("cat_to_cart") > 0) & (pl.col("cat") == 0), "cat_to_cart>0 ПРИ cat=0")
cnt((pl.col("cat_to_ord") > 0) & (pl.col("cat") == 0), "cat_to_ord>0 ПРИ cat=0")
cnt((pl.col("cat") == 1) & (pl.col("cat_to_cart") == 0) & (pl.col("cat_to_ord") == 0),
    "cat=1 но НИЧЕГО в каталоге (просмотр)")

print("\n=== из чего состоят строки без канала, но с активностью ===")
g = df.filter((pl.col("search") == 0) & (pl.col("cat") == 0) &
              ((pl.col("to_cart") > 0) | (pl.col("to_ord") > 0) | (pl.col("gmv") > 0)))
print(f"  всего {g.height:,}")
print("  из них с gmv_search>0:", g.filter(pl.col("gmv_search") > 0).height)
print("  из них с gmv_cat>0:   ", g.filter(pl.col("gmv_cat") > 0).height)
print("  из них с search_to_cart>0:", g.filter(pl.col("search_to_cart") > 0).height)
print("  из них с cat_to_cart>0:   ", g.filter(pl.col("cat_to_cart") > 0).height)

print("\n=== ТАКСОНОМИЯ ДНЯ (взаимоисключающие типы строк) ===")
types = {
 "A. пустая (ничего вообще)": (pl.col("searches")==0)&(pl.col("cat")==0)&(pl.col("to_cart")==0)&(pl.col("to_ord")==0),
 "B. только поиск, без корзины/заказа": (pl.col("searches")>0)&(pl.col("to_cart")==0)&(pl.col("to_ord")==0),
 "C. корзина без заказа": (pl.col("to_cart")>0)&(pl.col("to_ord")==0),
 "D. заказ": (pl.col("to_ord")>0),
}
prev = None
for lab, e in types.items():
    ex = e if prev is None else (e & ~prev)
    k = df.filter(ex).height
    print(f"  {lab:42s} {k:>11,}  ({100*k/n:5.2f}%)")
    prev = e if prev is None else (prev | e)
