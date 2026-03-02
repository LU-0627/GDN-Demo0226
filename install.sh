# pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html
# pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html
# pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html
# pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html
# pip install torch-geometric==1.5.0

# 安装torch-scatter（适配2.5.1+cu121，2.5.0的源兼容2.5.1）
pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

# 安装torch-sparse
pip install --no-index torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

# 安装torch-cluster
pip install --no-index torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

# 安装torch-spline-conv
pip install --no-index torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

# 安装兼容的torch-geometric（1.5.0版本太旧，建议升级到适配2.5.x的版本）
pip install torch-geometric  # 不指定1.5.0，自动安装兼容版本
# 若必须用1.5.0（代码强依赖），则执行：pip install torch-geometric==1.5.0