import os
import shutil
import glob
from pathlib import Path

def split_orc_files():
    """
    为CSCN-A0007到CSCN-A0016文件夹中的orc文件创建半天分割
    """
    # 基础路径 (当前目录，可以根据需要修改)
    base_path = "."
    
    # CSCN文件夹名称列表
    cscn_folders = [f"CSCN-A{str(i).zfill(4)}" for i in range(7, 17)]
    
    # 源文件夹和目标文件夹名称
    source_folder = "2025-06-29"
    target_folder_1 = "2025-06-29-half1"
    target_folder_2 = "2025-06-29-half2"
    
    processed_count = 0
    error_count = 0
    
    for cscn_folder in cscn_folders:
        try:
            print(f"正在处理文件夹: {cscn_folder}")
            
            # 构建完整路径
            cscn_path = Path(base_path) / cscn_folder
            source_path = cscn_path / source_folder
            target_path_1 = cscn_path / target_folder_1
            target_path_2 = cscn_path / target_folder_2
            
            # 检查CSCN文件夹是否存在
            if not cscn_path.exists():
                print(f"  警告: 文件夹 {cscn_folder} 不存在，跳过")
                continue
            
            # 检查源文件夹是否存在
            if not source_path.exists():
                print(f"  警告: 源文件夹 {source_path} 不存在，跳过")
                continue
            
            # 获取所有orc文件并排序
            orc_files = sorted(glob.glob(str(source_path / "*.orc")))
            
            if len(orc_files) == 0:
                print(f"  警告: 在 {source_path} 中未找到orc文件")
                continue
            elif len(orc_files) != 6:
                print(f"  警告: 在 {source_path} 中找到 {len(orc_files)} 个orc文件，期望6个")
            
            # 创建目标文件夹
            target_path_1.mkdir(exist_ok=True)
            target_path_2.mkdir(exist_ok=True)
            print(f"  创建文件夹: {target_folder_1}, {target_folder_2}")
            
            # 分割文件列表
            half_point = len(orc_files) // 2
            first_half = orc_files[:half_point]
            second_half = orc_files[half_point:]
            
            # 复制前一半文件到half1文件夹
            print(f"  复制前 {len(first_half)} 个文件到 {target_folder_1}")
            for file_path in first_half:
                file_name = os.path.basename(file_path)
                dest_path = target_path_1 / file_name
                shutil.copy2(file_path, dest_path)
                print(f"    复制: {file_name}")
            
            # 复制后一半文件到half2文件夹
            print(f"  复制后 {len(second_half)} 个文件到 {target_folder_2}")
            for file_path in second_half:
                file_name = os.path.basename(file_path)
                dest_path = target_path_2 / file_name
                shutil.copy2(file_path, dest_path)
                print(f"    复制: {file_name}")
            
            processed_count += 1
            print(f"  ✓ {cscn_folder} 处理完成\n")
            
        except Exception as e:
            error_count += 1
            print(f"  ✗ 处理 {cscn_folder} 时发生错误: {str(e)}\n")
    
    # 输出总结
    print("=" * 50)
    print(f"处理完成!")
    print(f"成功处理: {processed_count} 个文件夹")
    print(f"处理失败: {error_count} 个文件夹")
    print("=" * 50)

def verify_structure():
    """
    验证文件结构的辅助函数
    """
    base_path = "."
    cscn_folders = [f"CSCN-A{str(i).zfill(4)}" for i in range(7, 17)]
    
    print("验证文件结构:")
    print("-" * 30)
    
    for cscn_folder in cscn_folders:
        cscn_path = Path(base_path) / cscn_folder
        source_path = cscn_path / "2025-06-29"
        
        if cscn_path.exists():
            print(f"✓ {cscn_folder} 存在")
            if source_path.exists():
                orc_count = len(glob.glob(str(source_path / "*.orc")))
                print(f"  └─ 2025-06-29 存在，包含 {orc_count} 个orc文件")
            else:
                print(f"  └─ 2025-06-29 不存在")
        else:
            print(f"✗ {cscn_folder} 不存在")
    print()

if __name__ == "__main__":
    print("ORC文件分割脚本")
    print("=" * 50)
    
    # 首先验证文件结构
    verify_structure()
    
    # 询问用户是否继续
    response = input("是否继续执行分割操作？(y/n): ").lower().strip()
    if response in ['y', 'yes', '是', 'Y']:
        split_orc_files()
    else:
        print("操作已取消")