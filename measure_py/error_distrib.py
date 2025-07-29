import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.backends.backend_pdf as pdf_backend
import os
from pathlib import Path
from icecream import ic

img_width = 4032
img_height = 3036
# 箱型图、误差频率分布的误差的大范围和小范围
thresh_a1 = -1.1
thresh_a2 = 1.1
thresh_b1 = -3.1
thresh_b2 = 3.1
# 误差插值小范围显示的误差范围
thresh_c1 = -1.1
thresh_c2 = 1.1

# 定义长度范围的边界
value_bins = [0, 300, 500, 700, 1000, 1300, 1500, 1700, 2000, 2300, 2500, 2700, 3000,3300,3500]  # 根据实际情况调整范围边界
# value_bins = [0, 1000, 1500, 2000, 2500] 

# # 读取表格
# filename = '122cm_20240917_2x'  
# csv_file_path = os.path.join('/mnt/e/stereovision_camera/calibration_images/charuco_0912_2/' + filename +'.csv')  
# pdf_filename =  filename + '.pdf'
# pdf_filepath = os.path.join('/mnt/e/stereovision_camera/calibration_images/charuco_0912_2/', pdf_filename)

# 如本拼接
# 获取环境变量
# folder_name = os.environ.get('FOLDER_NAME')
# # samples_folder = "/mnt/e/ruben_test/ruben/0703/2.7_3/img"
# samples_folder = os.environ.get('FOLDER_PATH')
# csv_file_path = os.path.join(samples_folder, folder_name + '.csv')
# # 创建一个PDF文件
# pdf_filename = folder_name + '.pdf'
# pdf_filepath = os.path.join(samples_folder, pdf_filename)

# 双目
# 获取环境变量
folder_name = os.environ.get('FOLDER_NAME')
# folder_name = '20250519_hik_6mm_aftershake'
TARGET_FOLDER2 = Path().home() / 'DCIM'
csv_file_path = os.path.join(TARGET_FOLDER2, folder_name + '.csv')
# 创建一个PDF文件
pdf_filename = folder_name + '.pdf'
pdf_filepath = os.path.join(TARGET_FOLDER2, pdf_filename)

# 读取csv表格
df = pd.read_csv(csv_file_path)

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# generate a chart
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde

pdf1_filename = "plot.pdf"
pdf1 = pdf_backend.PdfPages(pdf1_filename)
###############################################################################################
# 绘制箱线图

# 将数据按照长度范围进行分组
df['value_range'] = pd.cut(df['true_value'], bins=value_bins)

# 创建一个空列表来存储每个长度范围内的数据
error_data1 = {}
error_data2 = {}
data_counts1 = {}  # 存储全部范围数据数量
data_counts2 = {}  # 存储小范围内的数据数量
withina = {}  
withinb = {}  

# 遍历每个长度范围
for value_range in df['value_range'].unique():
    # 选择当前长度范围内的数据
    errors1 = df.loc[df['value_range'] == value_range, 'error']
    # 选择当前长度范围内在小范围内的数据
    errors2 = df.loc[(df['value_range'] == value_range) & (df['error'].between(thresh_b1, thresh_b2)), 'error']
    # 将数据转换为NumPy数组
    errors1 = errors1.values
    errors2 = errors2.values
    # 将数据添加到字典中
    error_data1[value_range] = errors1.tolist()
    error_data2[value_range] = errors2.tolist()
    # 计算数据数量
    data_counts1[value_range] = len(errors1)
    data_counts2[value_range] = len(errors2)
    # 计算非离群值占比
    
    withina[value_range] = len(errors1[~((errors1 < thresh_a1) | (errors1 > thresh_a2))]) / len(errors1) * 100
    withinb[value_range] = len(errors1[~((errors1 < thresh_b1) | (errors1 > thresh_b2))]) / len(errors1) * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

import re
# 修改箱型图横坐标标签
xtick_labels = [str(label) for label in error_data1.keys()]
xtick_labels = [re.sub(r'(\d+)', lambda x: f"{float(x.group())/1000:.1f}", label) for label in xtick_labels]

# 绘制所有数据的箱线图
bp1 = ax1.boxplot(error_data1.values(), labels=xtick_labels, patch_artist=True)
# 设置更新后的x轴刻度标签
# ax1.set_xticklabels(xtick_labels)
ax1.set_xlabel('Length Range')
ax1.set_ylabel('Error')
ax1.set_title('Error Distribution by Length Range')

# 在每个箱线图上显示数据数量
for i, (box, value_range) in enumerate(zip(bp1['boxes'], error_data1.keys()), start=1):
    if len(error_data1[value_range]) > 0:
        max_value = max(error_data1[value_range])
    else:
        max_value = 0

    # ax1.text(i, max_value, f'n={data_counts1[value_range]}', ha='center', va='bottom')
    ax1.text(i, max_value, f'n={data_counts1[value_range]}\n%{thresh_a2}={withina[value_range]:.2f}%\n%{thresh_b2}={withinb[value_range]:.2f}%', ha='center', va='bottom')

# 修改箱型图横坐标标签
xtick_labels = [str(label) for label in error_data2.keys()]
xtick_labels = [re.sub(r'(\d+)', lambda x: f"{float(x.group())/1000:.1f}", label) for label in xtick_labels]

# 绘制在[-5, 5]范围内的数据的箱线图
bp2 = ax2.boxplot(error_data2.values(), labels=xtick_labels, patch_artist=True)
ax2.set_xlabel('Length Range')
ax2.set_ylabel('Error')
ax2.set_title('Error(' + str(thresh_b1) + ' to ' + str(thresh_b2) + 'mm) Distribution by Length Range')

# 在每个箱线图上显示数据数量
for i, (box, value_range) in enumerate(zip(bp2['boxes'], error_data2.keys()), start=1):
    if len(error_data2[value_range]) > 0:
        max_value = max(error_data2[value_range])
    else:
        max_value = 0

    ax2.text(i, max_value, f'n={data_counts2[value_range]}', ha='center', va='bottom')
    
# 调整子图之间的间距
# plt.subplots_adjust(hspace=0.5)
plt.tight_layout()
# fig.savefig(folder_path + '\\' + file + '_box_plot.png')
pdf1.savefig()
# plt.savefig(pdf1_filename, format='pdf')

######################################################################################
'''# 不同量程范围误差频率分布

# 获取唯一的 value_range 值列表
value_ranges = df['value_range'].unique()

# 定义子图的行数和列数
num_rows = 3
num_cols = np.ceil(len(value_ranges) / num_rows).astype(int)

# 创建子图
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

fig.tight_layout(pad=3.0)

# 遍历每个唯一的 value_range 值
for i, value_range in enumerate(value_ranges):
    # 选择当前 value_range 对应的数据
    errors = df.loc[df['value_range'] == value_range, 'error'].values.reshape(-1, 1)
    # if len(errors) < 2:
    #     continue

    # 获取当前子图的坐标    
    row = i // num_cols
    col = i % num_cols

    # 绘制直方图
    ax1 = axes[row, col]
    ax1.tick_params(axis='x', labelsize=6)  
    ax1.tick_params(axis='y', labelsize=6)
    counts, bins, patches = ax1.hist(errors, bins=20, edgecolor='black', label='Histogram')

    # 创建第二个坐标轴
    ax2 = ax1.twinx()
    ax2.tick_params(axis='y', labelsize=6)

    # # 使用Gaussian Mixture Model进行拟合并绘制
    # gmm = GaussianMixture(n_components=2, random_state=0).fit(errors)
    # x = np.linspace(errors.min(), errors.max(), 1000).reshape(-1, 1)
    # gmm_pdf = np.exp(gmm.score_samples(x))
    # ax2.plot(x, gmm_pdf, 'r-', label='GMM', linewidth=1)

    # # 使用Kernel Density Estimation进行拟合并绘制
    # kde = gaussian_kde(errors.ravel())
    # kde_pdf = kde(x.ravel())
    # ax2.plot(x, kde_pdf, 'k--', label='KDE', linewidth=1)

    # 设置图表标题和标签
    ax1.set_title(f'Value Range: {value_range}', fontsize=8)
    ax1.set_xlabel('Error (mm)', fontsize=8)
    ax1.set_ylabel('Frequency', fontsize=8)
    ax2.set_ylabel('Probability', fontsize=8)
    ax1.grid(True)

    # 添加图例
    # ax1.legend(loc='upper left', fontsize='x-small')
    # ax2.legend(loc='upper right', fontsize='x-small')

# 调整子图之间的间距
# plt.subplots_adjust(wspace=0.25, hspace=0.4)

pdf1.savefig()
# plt.savefig(pdf1_filename, format='pdf')
# 显示图表
# plt.show()

######################################################################################
# 全部范围与小范围的误差频率分布

fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 10))
errors = df['error'].values.reshape(-1, 1)  # 确保数据是二维的

# 绘制直方图
# fig, ax1 = plt.subplots(figsize=(10, 6))
counts, bins, patches = ax1.hist(errors, bins=100, edgecolor='black', label='Histogram')

# 创建第二个坐标轴
ax2 = ax1.twinx()

# 使用Gaussian Mixture Model进行拟合并绘制
gmm = GaussianMixture(n_components=2, random_state=0).fit(errors)
x = np.linspace(errors.min(), errors.max(), 1000).reshape(-1, 1)
gmm_pdf = np.exp(gmm.score_samples(x))
ax2.plot(x, gmm_pdf, 'r-', label='GMM', linewidth=2)

# 使用Kernel Density Estimation进行拟合并绘制
kde = gaussian_kde(errors.ravel())
kde_pdf = kde(x.ravel())
ax2.plot(x, kde_pdf, 'k--', label='KDE', linewidth=2)

# 设置图表标题和标签
ax1.set_xlabel('Error (mm)')
ax1.set_ylabel('Frequency')
ax2.set_ylabel('Probability')
plt.title('Error Distribution Histogram with GMM and KDE Fits')
ax1.grid(True)

# 添加图例
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')


# remove errors that are greater than 5mm or less than -5mm
errors_b = errors[(errors >= thresh_b1) & (errors <= thresh_b2)]
errors_b = errors_b.reshape(-1, 1)  # 确保数据是二维的


counts, bins, patches = ax3.hist(errors_b, bins=100, edgecolor='black', label='Histogram')


# 创建第二个坐标轴
ax4 = ax3.twinx()

# # 使用Gaussian Mixture Model进行拟合并绘制
# gmm = GaussianMixture(n_components=2, random_state=0).fit(errors_b)
# x = np.linspace(errors_b.min(), errors_b.max(), 1000).reshape(-1, 1)
# gmm_pdf = np.exp(gmm.score_samples(x))
# ax4.plot(x, gmm_pdf, 'r-', label='GMM', linewidth=2)

# # 使用Kernel Density Estimation进行拟合并绘制
# kde = gaussian_kde(errors_b.ravel())
# kde_pdf = kde(x.ravel())
ax4.plot(x, kde_pdf, 'k--', label='KDE', linewidth=2)

# 设置图表标题和标签
ax3.set_xlabel('Error (mm)')
ax3.set_ylabel('Frequency')
ax4.set_ylabel('Probability')
plt.title('Error(' + str(thresh_b1) + ' to ' + str(thresh_b2) + 'mm)  Distribution Histogram with GMM and KDE Fits')
ax3.grid(True)

# 添加图例
ax3.legend(loc='upper left')
ax4.legend(loc='upper right')


pdf1.savefig()
# plt.savefig(pdf1_filename, format='pdf')
# 显示图表
# plt.show()'''

#####################################################################################################################
# # 误差插值

# # 计算error的绝对值
# # df['error_abs'] = df['error'].abs()

# # 创建新的空表
# new_df = pd.DataFrame(columns=['coord', 'error_abs', 'true_value', 'filename', 'id'])

# # 将coord1列和error_abs列的值添加到新表
# new_df['coord'] = df['coord1_left']
# new_df['error_abs'] = df['error_abs']
# new_df['true_value'] = df['true_value']
# new_df['filename'] = df['filename']
# new_df['id'] = df['id1']

# # 创建临时表，存储coord2列和error_abs列的值
# temp_df = pd.DataFrame(columns=['coord', 'error_abs', 'true_value', 'filename', 'id'])
# temp_df['coord'] = df['coord2_left']
# temp_df['error_abs'] = df['error_abs']
# temp_df['true_value'] = df['true_value']
# temp_df['filename'] = df['filename']
# temp_df['id'] = df['id2']

# # 将临时表添加到新表中
# new_df = pd.concat([new_df, temp_df], ignore_index=True)


# new_df['coord_x'] = new_df['coord'].str.replace('[\[\]]', '', regex=True).str.split().str[0].astype(float)
# new_df['coord_y'] = new_df['coord'].str.replace('[\[\]]', '', regex=True).str.split().str[1].astype(float)
# new_df.drop('coord', axis=1, inplace=True)
# # new_df.to_csv(folder_path + '\\' + file + '_result01.csv', index=False)

# result2_df = pd.DataFrame(columns=['coord_x', 'coord_y', 'error_mean'])


# # 对 new_df 进行分组计算均值
# grouped = new_df.groupby('coord_x')['error_abs'].mean().reset_index()

# # 将分组结果的值存储到新表中
# result2_df['coord_x'] = grouped['coord_x']
# result2_df['error_mean'] = grouped['error_abs']

# # 根据 coord_x 对应的值查找对应的 coord_y
# for index, row in result2_df.iterrows():
#     coord_x_value = row['coord_x']
#     coord_y_value = new_df.loc[new_df['coord_x'] == coord_x_value, 'coord_y'].values[0]
#     result2_df.at[index, 'coord_y'] = coord_y_value

# # result2_df.to_csv(folder_path + '\\' + file + '_result02.csv', index=False)
# # result2_df.to_csv(folder_path + '\\' + file + '_' + model + '_result.csv', index=False)

# # 创建一个图形对象

# grid_x, grid_y = np.mgrid[0:img_width:500j, 0:img_height:500j]
# X, Y = grid_x, grid_y

# ranges = [
#     (None, 1000),      # 小于1000
#     (1000, 1500),      # 1000-1500
#     (1500, 2000),      # 1500-2000
#     (2000, None)       # 大于2000
# ]
# range_labels = [
#     'true_value < 1000',
#     '1000 ≤ true_value < 1500',
#     '1500 ≤ true_value < 2000',
#     'true_value ≥ 2000'
# ]

# fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# axes = axes.flatten()

# for idx, (low, high) in enumerate(ranges):
#     if low is None:
#         mask = new_df['true_value'] < high
#     elif high is None:
#         mask = new_df['true_value'] >= low
#     else:
#         mask = (new_df['true_value'] >= low) & (new_df['true_value'] < high)
#     sub_df = new_df[mask]
#     # 分组计算均值
#     grouped = sub_df.groupby(['coord_x', 'coord_y'])['error_abs'].mean().reset_index()
#     points = grouped[['coord_x', 'coord_y']].values
#     values = grouped['error_abs'].values

#     if len(points) < 1:
#         continue  # 跳过点数太少的区间

#     # 网格插值
#     Z_interpolated = griddata(points, values, (X, Y), method='linear')
#     vmin = np.nanmin(values)
#     vmax = np.nanmax(values)
#     norm = Normalize(vmin=vmin, vmax=vmax)
#     ax = axes[idx]
#     im = ax.imshow(Z_interpolated.T, cmap='viridis', extent=[0, img_width, 0, img_height], origin='lower', norm=norm)
#     ax.set_title(range_labels[idx])
#     ax.set_xlim(0, img_width)
#     ax.set_ylim(0, img_height)
#     ax.invert_yaxis()
#     cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     cbar.ax.set_ylabel('error_mean')

    # ----------- 标注 id1（红色，左侧）和 id2（蓝色，右侧） -----------
    # 1. 找到当前范围内所有原始行的索引
    # if low is None:
    #     mask_raw = (df['true_value'] < high)
    # elif high is None:
    #     mask_raw = (df['true_value'] >= low)
    # else:
    #     mask_raw = (df['true_value'] >= low) & (df['true_value'] < high)

    # # id1
    # ids1 = df.loc[mask_raw, ['id1', 'coord1_left']].drop_duplicates()
    # for _, row in ids1.iterrows():
    #     id_val = row['id1']
    #     coord_str = row['coord1_left']
    #     # if str(id_val) == '23':
    #     # ic(id_val, coord_str)
    #     try:
    #         nums = coord_str.replace('[', '').replace(']', '').split()
    #         x = float(nums[0])
    #         y = float(nums[1])
    #         ax.scatter(x, y, color='red', s=40, marker='o')
    #         ax.text(x-30, y, str(id_val), color='red', fontsize=10, ha='right', va='center', fontweight='bold')
    #         # if str(id_val) == '23':
    #         # ic(id_val,x,y)
    #     except Exception:
    #         continue

    # # id2
    # ids2 = df.loc[mask_raw, ['id2', 'coord2_left']].drop_duplicates()
    # for _, row in ids2.iterrows():
    #     id_val = row['id2']
    #     coord_str = row['coord2_left']
    #     try:
    #         nums = coord_str.replace('[', '').replace(']', '').split()
    #         x = float(nums[0])
    #         y = float(nums[1])
    #         ax.scatter(x, y, color='blue', s=40, marker='o')
    #         ax.text(x+30, y, str(id_val), color='blue', fontsize=10, ha='left', va='center', fontweight='bold')
    #     except Exception:
    #         continue

plt.tight_layout()
pdf1.savefig()
pdf1.close()

#####################################################################################################################

# 创建统计数据表
stats_data = pd.DataFrame(columns=['value_range', 'count', 'mean', 'std', 'median', 'q1', 'q3', 'min', 'max', 'min_outlier', 'max_outlier'])

# 按量程范围统计数据
for value_range in df['value_range'].unique():
    subset = df[df['value_range'] == value_range]
    error_abs = subset['error_abs']
    error = subset['error']
    lower_bound = value_range.left / 1000
    upper_bound = value_range.right / 1000

    value_range_m = f"({lower_bound}, {upper_bound}]"

    # 计算统计值
    count = subset.shape[0]
    mean = error_abs.mean()
    std = error_abs.std()
    median = error.median()
    q1 = error.quantile(0.25)
    q3 = error.quantile(0.75)
    iqr = q3 -q1
    min_outlier = error.min()
    max_outlier = error.max()
    min = q1 - 1.5*iqr
    max = q3 + 1.5*iqr
    if min_outlier >= min:
        min = min_outlier
    if max_outlier <= max:
        max = max_outlier

    # 将统计值添加到新的数据表
    stats_row = pd.DataFrame({
        'count': [count],
        'value_range': [value_range_m],
        'mean': [mean],
        'std': [std],
        'median': [median],
        'q1': [q1],
        'q3': [q3],
        'min': [min],
        'max': [max],
        'min_outlier': [min_outlier],
        'max_outlier': [max_outlier]
    })
    stats_data = pd.concat([stats_data, stats_row])

# 重置索引
stats_data = stats_data.reset_index(drop=True)
error_within2 = df[df['error_abs'] < 2.1]
count_error_within2 = error_within2.shape[0]


# 添加总计行
total_count = df.shape[0]
overall_mean = df['error_abs'].mean()
overall_std = df['error_abs'].std()
overall_median = df['error'].median()
overall_q1 = df['error'].quantile(0.25)
overall_q3 = df['error'].quantile(0.75)
overall_iqr = overall_q3 -overall_q1
overall_min_outlier = df['error'].min()
overall_max_outlier = df['error'].max()
overall_min = overall_q1 - 1.5*overall_iqr
overall_max = overall_q3 + 1.5*overall_iqr

percent_all_within2 = (count_error_within2/total_count)*100
print("误差不超过2mm的数据占比:", percent_all_within2,"%")

if overall_min_outlier >= overall_min:
    overall_min = overall_min_outlier
if overall_max_outlier <= overall_max:
    overall_max = overall_max_outlier

total_row = pd.DataFrame({
    'value_range': 'total',
    'count': total_count,
    'mean': overall_mean,
    'std': overall_std,
    'median': overall_median,
    'q1': overall_q1,
    'q3': overall_q3,
    'min': overall_min,
    'max': overall_max,
    'min_outlier': overall_min_outlier,
    'max_outlier': overall_max_outlier
}, index=[0])

stats_data = pd.concat([stats_data, total_row], ignore_index=True)
        
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# 转换DataFrame中的数据类型为字符串
stats_data = stats_data.astype(str)

# 设置样式
styles = getSampleStyleSheet()
cell_style = styles["BodyText"]
cell_style.alignment = 1  # 居中对齐

# 创建PDF文件
pdf_file_table = 'table.pdf'
doc = SimpleDocTemplate(pdf_file_table, pagesize=letter)

# 创建表格
table_data = [stats_data.columns.tolist()]  # 添加表头
table_data += stats_data.values.tolist()  # 添加数据行
# num_cols = len(table_data[0])  # 获取列数
col_width = 50  # 设置列宽

# 将具有两位小数的数据格式化为字符串
for i in range(1, len(table_data)):
    for j in range(len(table_data[i])):
        if j > 1:
            value = float(table_data[i][j])
            formatted_value = "{:.2f}".format(value)
            table_data[i][j] = formatted_value

# 添加单元格内容
for i, row in enumerate(table_data):
    for j, value in enumerate(row):
        cell = Paragraph(value, cell_style)
        table_data[i][j] = cell
        
table = Table(table_data)
table._argW[1:] = [col_width] * (len(table_data[0]) - 1)
# 设置表格样式
table.setStyle(TableStyle([
    # 表头样式
    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 12),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    # 数据行样式
    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
]))

# 将表格添加到文档中
elements = []
elements.append(table)

# 添加表例
from reportlab.lib.styles import getSampleStyleSheet

styles = getSampleStyleSheet()
custom_style = styles["Normal"]
custom_style.fontSize = 14
custom_style.leading = 20  # 行间距

paragraph = Paragraph("<br/>std: standard deviation<br/>q1: 1st quartile <br/>q3: 3rd quartile<br/>min: minimum excluding outliers<br/>max: maximum excluding outliers<br/>min_outlier: minimum including outliers<br/>max_outlier: maximum including outliers", 
                      custom_style)
elements.append(paragraph)
doc.build(elements)


from PyPDF2 import PdfReader, PdfWriter
pdf1 = PdfReader(pdf1_filename, "rb")
pdf2_filename = "table.pdf"
pdf2 = PdfReader(pdf2_filename, "rb")

# 创建新的PDF写入对象
# output_pdf_filename = "output.pdf"
output_pdf = PdfWriter(pdf_filepath)

# 将第一个PDF的前两页添加到输出PDF
for page_num in range(0, 1):
    output_pdf.add_page(pdf1.pages[page_num])

# 将第二个PDF的页面添加到输出PDF
for page in pdf2.pages:
    output_pdf.add_page(page)

# 将第一个PDF的剩余页面添加到输出PDF
for page_num in range(1, len(pdf1.pages)):
    output_pdf.add_page(pdf1.pages[page_num])

# 保存输出PDF文件
with open(pdf_filepath, "wb") as f:
    output_pdf.write(f)
os.remove(pdf1_filename)
os.remove(pdf2_filename)