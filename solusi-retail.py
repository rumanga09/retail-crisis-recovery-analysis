import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_excel(os.path.join(BASE_DIR, 'data_penjualan.xlsx'))

df.columns = df.columns.str.lower()

df['tgl_transaksi'] = pd.to_datetime(df['tgl_transaksi'])


penjualan = (
    df.groupby([
        'tgl_transaksi',
        'kode_produk',
        'nama_produk'
    ])['total_nilai']
    .sum()
    .reset_index()
    .sort_values(['kode_produk', 'tgl_transaksi'])
)


penjualan['ma_3'] = (
    penjualan.groupby('kode_produk')['total_nilai']
    .transform(
        lambda x: x.rolling(3, min_periods=3).mean()
    )
)


penjualan['is_up'] = (
    penjualan.groupby('kode_produk')['ma_3']
    .diff()
    .gt(0)
)

penjualan['group'] = (
    penjualan.groupby('kode_produk')['is_up']
    .transform(lambda x: (~x).cumsum())
)


tren_naik = (
    penjualan[penjualan['is_up']]
    .groupby([
        'kode_produk',
        'nama_produk',
        'group'
    ])
    .agg(
        start_date=('tgl_transaksi', 'min'),
        end_date=('tgl_transaksi', 'max'),
        consecutive_days=('tgl_transaksi', 'count'),
        start_ma=('ma_3', 'first'),
        end_ma=('ma_3', 'last')
    )
    .reset_index()
)


tren_naik = tren_naik[
    tren_naik['consecutive_days'] >= 12
].copy()


tren_naik['growth_pct'] = (
    (tren_naik['end_ma'] - tren_naik['start_ma'])
    / tren_naik['start_ma']
) * 100


tren_naik = tren_naik.sort_values(
    'growth_pct',
    ascending=False
)


produk_rising_star_kode = (
    tren_naik['kode_produk']
    .unique()
    .tolist()
)

produk_rising_star_nama = (
    tren_naik['nama_produk']
    .unique()
    .tolist()
)


produk_top_3 = (
    df.groupby([
        'kode_produk',
        'nama_produk'
    ])['total_nilai']
    .sum()
    .sort_values(ascending=False)
    .head(3)
    .reset_index()['kode_produk']
    .tolist()
)


produk_visual = produk_top_3 + [
    p for p in produk_rising_star_kode
    if p not in produk_top_3
]


data_visual = penjualan[
    penjualan['kode_produk'].isin(produk_visual)
].copy()


base_map = (
    data_visual
    .groupby('kode_produk')['ma_3']
    .apply(lambda x: x.dropna().iloc[0] if not x.dropna().empty else None)
    .to_dict()
)


data_visual['index_growth'] = (
    data_visual['ma_3']
    / data_visual['kode_produk'].map(base_map)
) * 100


nama_map = (
    data_visual.groupby('kode_produk')['nama_produk']
    .first()
    .to_dict()
)


sorted_report = tren_naik.sort_values(
    by='growth_pct',
    ascending=False
)


custom_palette = [
    '#FFD700',
    '#C0C0C0',
    '#CD7F32',
    '#2ecc71',
    '#3498db',
    '#9b59b6',
    '#e74c3c',
    '#34495e'
]

default_color = '#95a5a6'

mapping_warna = {}
mapping_rank = {}


for i, row in enumerate(sorted_report.itertuples()):

    kode_produk = row.kode_produk

    mapping_warna[kode_produk] = (
        custom_palette[i]
        if i < len(custom_palette)
        else default_color
    )

    mapping_rank[kode_produk] = i + 1


grey_colors = [
    '#B0B0B0',
    '#909090',
    '#707070'
]


font_title = {
    'family': 'sans-serif',
    'color': 'black',
    'weight': 'bold',
    'size': 16
}

font_label = {
    'family': 'sans-serif',
    'weight': 'normal',
    'size': 12
}


fig = plt.figure(figsize=(15, 8), dpi=100)
ax = fig.add_subplot(111)


for idx, produk in enumerate(sorted(produk_top_3)):

    grup = data_visual[
        data_visual['kode_produk'] == produk
    ]

    ax.plot(
        grup['tgl_transaksi'],
        grup['index_growth'],
        linestyle='--',
        linewidth=2,
        marker='o',
        markersize=3,
        color=grey_colors[idx],
        alpha=0.7,
        label=f"Top Sales: {nama_map.get(produk, produk)}"
    )


for produk in sorted(produk_rising_star_kode):

    grup = data_visual[
        data_visual['kode_produk'] == produk
    ]

    warna = mapping_warna.get(
        produk,
        default_color
    )

    rank = mapping_rank.get(
        produk,
        '?'
    )

    ax.plot(
        grup['tgl_transaksi'],
        grup['index_growth'],
        marker='o',
        markersize=4,
        linewidth=2.5,
        color=warna,
        label=f"Rank {rank}: {nama_map.get(produk, produk)}"
    )


ax.set_title(
    'ANALISIS PERTUMBUHAN RELATIF PRODUK RISING STAR\n'
    '(Dengan Benchmark Top 3 Total Penjualan)',
    fontdict=font_title,
    pad=20
)

ax.set_xlabel(
    'Periode Tanggal',
    fontdict=font_label,
    labelpad=10
)

ax.set_ylabel(
    'Indeks Pertumbuhan (Base 100)',
    fontdict=font_label,
    labelpad=10
)

ax.grid(
    True,
    linestyle='--',
    linewidth=0.5,
    alpha=0.5
)

ax.axhline(
    y=100,
    color='black',
    linestyle='-',
    linewidth=1,
    alpha=0.5
)

plt.xticks(
    rotation=45,
    ha='right',
    fontsize=10
)

plt.yticks(fontsize=10)

handles, labels = ax.get_legend_handles_labels()

top_items = []
rising_items = []

for h, l in zip(handles, labels):

    if l.startswith('Top Sales'):
        top_items.append((h, l))
    else:
        rising_items.append((h, l))


rising_items = sorted(
    rising_items,
    key=lambda x: int(
        x[1].split(':')[0].split()[1]
    )
)

legend_final = top_items + rising_items

handles_final = [x[0] for x in legend_final]
labels_final = [x[1] for x in legend_final]


ax.legend(
    handles_final,
    labels_final,
    title='Kategori Produk',
    title_fontsize=12,
    fontsize=10,
    bbox_to_anchor=(1.02, 1),
    loc='upper left',
    borderaxespad=0,
    frameon=True,
    shadow=True
)

plt.tight_layout()

plt.savefig(
    os.path.join(BASE_DIR, 'rising_star_index.png'),
    bbox_inches='tight'
)

plt.close()


fig2 = plt.figure(figsize=(15, 8), dpi=100)
ax2 = fig2.add_subplot(111)


for idx, produk in enumerate(sorted(produk_top_3)):

    grup = data_visual[
        data_visual['kode_produk'] == produk
    ]

    ax2.plot(
        grup['tgl_transaksi'],
        grup['total_nilai'],
        linestyle='--',
        linewidth=2,
        marker='o',
        markersize=3,
        color=grey_colors[idx],
        alpha=0.7,
        label=f"Top Sales: {nama_map.get(produk, produk)}"
    )


for produk in sorted(produk_rising_star_kode):

    grup = data_visual[
        data_visual['kode_produk'] == produk
    ]

    warna = mapping_warna.get(
        produk,
        default_color
    )

    rank = mapping_rank.get(
        produk,
        '?'
    )

    ax2.plot(
        grup['tgl_transaksi'],
        grup['total_nilai'],
        marker='o',
        markersize=4,
        linewidth=2.5,
        color=warna,
        label=f"Rank {rank}: {nama_map.get(produk, produk)}"
    )


ax2.set_title(
    'ANALISIS NILAI PENJUALAN PRODUK RISING STAR\n'
    '(Nilai Penjualan Asli)',
    fontdict=font_title,
    pad=20
)

ax2.set_xlabel(
    'Periode Tanggal',
    fontdict=font_label,
    labelpad=10
)

ax2.set_ylabel(
    'Total Nilai Penjualan',
    fontdict=font_label,
    labelpad=10
)

ax2.grid(
    True,
    linestyle='--',
    linewidth=0.5,
    alpha=0.5
)

plt.xticks(
    rotation=45,
    ha='right',
    fontsize=10
)

plt.yticks(fontsize=10)

handles2, labels2 = ax2.get_legend_handles_labels()

top_items2 = []
rising_items2 = []

for h, l in zip(handles2, labels2):

    if l.startswith('Top Sales'):
        top_items2.append((h, l))
    else:
        rising_items2.append((h, l))


rising_items2 = sorted(
    rising_items2,
    key=lambda x: int(
        x[1].split(':')[0].split()[1]
    )
)

legend_final2 = top_items2 + rising_items2

handles_final2 = [x[0] for x in legend_final2]
labels_final2 = [x[1] for x in legend_final2]


ax2.legend(
    handles_final2,
    labels_final2,
    title='Kategori Produk',
    title_fontsize=12,
    fontsize=10,
    bbox_to_anchor=(1.02, 1),
    loc='upper left',
    borderaxespad=0,
    frameon=True,
    shadow=True
)

plt.tight_layout()

plt.savefig(
    os.path.join(BASE_DIR, 'rising_star_actual.png'),
    bbox_inches='tight'
)

plt.close()


transaksi = (
    df.groupby('nomor_struk')['nama_produk']
    .apply(list)
    .tolist()
)


encoder = TransactionEncoder()

encoded = encoder.fit(transaksi).transform(transaksi)

keranjang = pd.DataFrame(
    encoded,
    columns=encoder.columns_
)


itemset_frekuen = apriori(
    keranjang,
    min_support=0.01,
    use_colnames=True
)


aturan = association_rules(
    itemset_frekuen,
    num_itemsets=len(keranjang),
    metric='lift',
    min_threshold=1
)


aturan = aturan[
    aturan['lift'] >= 2
].copy()


aturan['antecedents_str'] = aturan['antecedents'].apply(
    lambda x: ', '.join(sorted(list(x), reverse=True))
)

aturan['consequents_str'] = aturan['consequents'].apply(
    lambda x: ', '.join(sorted(list(x), reverse=True))
)


aturan = aturan[
    aturan['antecedents_str'].apply(
        lambda x: any(
            prod in x
            for prod in produk_rising_star_nama
        )
    )
    |
    aturan['consequents_str'].apply(
        lambda x: any(
            prod in x
            for prod in produk_rising_star_nama
        )
    )
]


aturan = aturan.sort_values(
    ['lift', 'support', 'confidence'],
    ascending=False
)


total_penjualan = (
    df.groupby('kode_produk')['total_nilai']
    .sum()
    .to_dict()
)


output_rising_star = tren_naik[[
    'kode_produk',
    'nama_produk',
    'growth_pct'
]].copy()


output_rising_star['total_penjualan'] = (
    output_rising_star['kode_produk']
    .map(total_penjualan)
)


output_rising_star.columns = [
    'Kode Produk',
    'Nama Produk',
    'Growth %',
    'Total Penjualan'
]


output_rising_star['Growth %'] = (
    output_rising_star['Growth %']
    .round(2)
)

output_rising_star['Total Penjualan'] = (
    output_rising_star['Total Penjualan']
    .round(0)
)


total_transaksi = df['nomor_struk'].nunique()


aturan['Jumlah Invoice'] = (
    aturan['support'] * total_transaksi
).round().astype(int)


output_packaging = aturan[[
    'antecedents_str',
    'consequents_str',
    'Jumlah Invoice',
    'support',
    'confidence',
    'lift'
]].copy()


output_packaging.columns = [
    'Jika Membeli',
    'Maka Membeli',
    'Jumlah Invoice',
    'Support',
    'Confidence',
    'Lift'
]


output_packaging['Support'] = (
    output_packaging['Support']
    .round(2)
)

output_packaging['Confidence'] = (
    output_packaging['Confidence']
    .round(2)
)

output_packaging['Lift'] = (
    output_packaging['Lift']
    .round(2)
)


with pd.ExcelWriter(
    os.path.join(BASE_DIR, 'retail_insight.xlsx'),
    engine='openpyxl'
) as writer:

    output_rising_star.to_excel(
        writer,
        sheet_name='Rising Star',
        index=False
    )

    output_packaging.to_excel(
        writer,
        sheet_name='Potential Packaging',
        index=False
    )


print('retail_insight.xlsx berhasil dibuat')
print('rising_star_index.png berhasil dibuat')
print('rising_star_actual.png berhasil dibuat')