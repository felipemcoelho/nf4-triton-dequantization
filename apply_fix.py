#!/usr/bin/env python3
"""
Apply fix to the kernel.py file after cloning the repository
Run this after cloning and before installing
"""

import sys

def fix_kernel():
    """Fix the kernel.py file to work on Tesla T4"""
    
    # Read the original file
    with open('nf4_triton_dequantization/kernel.py', 'r') as f:
        content = f.read()
    
    # Find and replace the problematic section
    old_code = """    # Store interleaved results using unrolled loop
    for i in tl.static_range(32):
        idx = i * 2
        if col_start + idx < n:
            tl.store(output_ptr + output_base + idx, low_scaled[i])
        if col_start + idx + 1 < n:
            tl.store(output_ptr + output_base + idx + 1, high_scaled[i])"""
    
    new_code = """    # Vectorized interleaved store - FIXED for T4
    # Store low values at even positions
    low_indices = offsets * 2
    low_mask = (col_start + low_indices) < n
    tl.store(output_ptr + output_base + low_indices, low_scaled, mask=low_mask)
    
    # Store high values at odd positions  
    high_indices = offsets * 2 + 1
    high_mask = (col_start + high_indices) < n
    tl.store(output_ptr + output_base + high_indices, high_scaled, mask=high_mask)"""
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        print("✅ Fixed tl.static_range issue")
    else:
        print("⚠️ Code pattern not found, checking if already fixed...")
        if "Vectorized interleaved store" in content:
            print("✅ Already fixed")
            return
        else:
            print("❌ Could not apply fix - manual intervention needed")
            return
    
    # Add PyTorch fallback if not present
    if "def pure_torch_fallback" not in content:
        fallback_code = '''

def pure_torch_fallback(module):
    """Pure PyTorch fallback for Tesla T4 and other GPUs where Triton is slow."""
    import torch
    
    weight = module.weight
    quant_state = weight.quant_state
    
    qweight = weight.data
    absmax = quant_state.absmax
    absmax32 = quant_state.state2.absmax
    dtype = quant_state.dtype
    device = qweight.device
    
    m = module.out_features
    n = module.in_features
    
    # NF4 lookup table
    nf4_table = torch.tensor([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ], dtype=torch.float32, device=device)
    
    blocks_per_row = (n + 63) // 64
    absmax32_per_row = (blocks_per_row + 3) // 4
    
    # Reshape scales
    if absmax.dim() == 1:
        if absmax.numel() == blocks_per_row:
            absmax = absmax.unsqueeze(0).expand(m, -1)
        elif absmax.numel() == m * blocks_per_row:
            absmax = absmax.view(m, blocks_per_row)
    
    if absmax32.dim() == 1:
        if absmax32.numel() == absmax32_per_row:
            absmax32 = absmax32.unsqueeze(0).expand(m, -1)
        elif absmax32.numel() == m * absmax32_per_row:
            absmax32 = absmax32.view(m, absmax32_per_row)
    
    # Convert to float
    absmax = absmax.float()
    absmax32 = absmax32.float()
    
    # Allocate output
    output = torch.zeros((m, n), dtype=dtype, device=device)
    
    # Process row by row
    for row in range(m):
        row_data = qweight[row]
        
        # Extract nibbles
        low = (row_data & 0xF).long()
        high = ((row_data >> 4) & 0xF).long()
        
        # Lookup values
        low_vals = nf4_table[low]
        high_vals = nf4_table[high]
        
        # Create interleaved row
        row_out = torch.zeros(n, dtype=torch.float32, device=device)
        row_out[0::2] = low_vals[:n//2]
        row_out[1::2] = high_vals[:n//2]
        
        # Apply scales
        for blk in range(blocks_per_row):
            start = blk * 64
            end = min(start + 64, n)
            scale = absmax[row, blk] * 0.00787401574803149606 * absmax32[row, blk // 4]
            row_out[start:end] *= scale
        
        output[row] = row_out.to(dtype)
    
    return output'''
        
        content += fallback_code
        print("✅ Added PyTorch fallback function")
    
    # Write the fixed file
    with open('nf4_triton_dequantization/kernel.py', 'w') as f:
        f.write(content)
    
    print("✅ File successfully patched")

if __name__ == "__main__":
    fix_kernel()