import graphviz
import os

def draw_architecture_diagram_hd():
    # 建立有向圖 (High Definition)
    dot = graphviz.Digraph('MODWT-MoE_Architecture_HD', comment='Production Model')
    
    # ==================== 全域視覺設定 (關鍵優化) ====================
    # dpi='300': 高解析度，解決字體糊掉的問題
    # nodesep='0.8': 同一層級節點的間距 (垂直距離)
    # ranksep='1.2': 不同層級節點的間距 (水平距離)
    # splines='ortho': 折線風格 (比較整齊)
    dot.attr(rankdir='LR', dpi='300', size='20,12', 
             nodesep='0.8', ranksep='1.2', splines='ortho', 
             bgcolor='white', compound='true')
    
    dot.attr('node', shape='box', style='filled', fontname='Arial', fontsize='12', margin='0.2,0.1')
    dot.attr('edge', penwidth='1.2', arrowsize='0.8', fontname='Arial', fontsize='10')

    # ==================== 1. 數據輸入與預處理 ====================
    with dot.subgraph(name='cluster_data') as c:
        c.attr(label='Data Preparation', color='grey', style='dashed', fontcolor='grey')
        c.attr('node', fillcolor='#E3F2FD', color='#90CAF9') # Light Blue
        
        c.node('RawData', '<<B>Raw Data</B><BR/>(USD_TWD.csv)>')
        c.node('FeatEng', '<<B>Feature Eng.</B><BR/>LogRet, Volatility, RSI<BR/>MACD, Accel>')
        c.node('Scaler', '<<B>Scaler</B><BR/>(Z-Score Norm)>')
        
        c.edge('RawData', 'FeatEng')
        c.edge('FeatEng', 'Scaler')

    # ==================== 2. 小波分解 (MODWT) ====================
    with dot.subgraph(name='cluster_modwt') as c:
        c.attr(label='Signal Decomposition', color='#FFB74D', style='rounded', fontcolor='#EF6C00')
        c.attr('node', fillcolor='#FFF3E0', color='#FFCC80') # Light Orange
        
        c.node('MODWT', '<<B>MODWT Algo</B><BR/>(db4, Level 4)>')
        
        # 使用 record 形狀來整齊排列分量
        c.node('Components', 
               '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">'
               '<TR><TD BGCOLOR="#FFE0B2"><B>Trend (cA4)</B></TD></TR>'
               '<TR><TD BGCOLOR="#FFE0B2"><B>Cyclic (cD4, cD3)</B></TD></TR>'
               '<TR><TD BGCOLOR="#FFE0B2"><B>High-Freq (cD2, cD1)</B></TD></TR>'
               '</TABLE>>', shape='plaintext', fillcolor='none')

        c.edge('Scaler', 'MODWT', label=' Volatility Signal')
        c.edge('MODWT', 'Components')

    # ==================== 3. 模型核心 (Model Core) ====================
    with dot.subgraph(name='cluster_model') as c:
        c.attr(label='Heterogeneous Hybrid Architecture (MODWT-MoE)', color='black', penwidth='2.0')
        
        # --- 3A. Base Branch (LSTM) ---
        with dot.subgraph(name='cluster_base') as base:
            base.attr(label='Base Branch (Anchor)', color='#66BB6A', style='filled', fillcolor='#E8F5E9')
            base.attr('node', fillcolor='#A5D6A7') # Green
            
            base.node('RawInput', '<<B>Raw Input</B><BR/>[Vol, RSI, Ret]>', fillcolor='white')
            base.node('LSTM', '<<B>RawLSTM</B><BR/>(2 Layers, Hidden=64)>')
            base.node('BaseDelta', '<<B>Base Prediction</B><BR/>(Δ base)>', shape='ellipse', style='filled', fillcolor='white')
            
            base.edge('RawInput', 'LSTM')
            base.edge('LSTM', 'BaseDelta')

        # --- 3B. Expert Branch (MoE) ---
        with dot.subgraph(name='cluster_experts') as exp:
            exp.attr(label='Expert Branch (Residual Correction)', color='#EF5350', style='filled', fillcolor='#FFEBEE')
            exp.attr('node', fillcolor='#FFCDD2') # Red
            
            # Context
            exp.node('Context', '<<B>Context Vector</B><BR/>Size=13<BR/>[Wavelets + Raw]>', fillcolor='#E0E0E0', color='#9E9E9E')

            # Experts (Parallel)
            exp.node('Exp1', '<<B>Trend Expert</B><BR/>(GRU + Attn)>')
            exp.node('Exp2', '<<B>Cyclic Expert</B><BR/>(GRU + Attn)>')
            exp.node('Exp3', '<<B>High-Freq Expert</B><BR/>(GRU + Attn)>')
            
            # Gating Mechanisms
            exp.node('Gating', '<<B>Gating Network</B><BR/>(Softmax)>', fillcolor='#AB47BC', fontcolor='white')
            exp.node('MetaGate', '<<B>Meta-Gate (α)</B><BR/>(Sigmoid)>', fillcolor='#5E35B1', fontcolor='white')
            
            # Aggregation
            exp.node('MoESum', '<<B>MoE Sum</B><BR/>Σ (w * Expert)>', shape='diamond', style='filled', fillcolor='white')

            # Edges inside
            exp.edge('Context', 'Gating')
            exp.edge('Context', 'MetaGate')
            
            # Invisible edges to align experts vertically
            exp.edge('Exp1', 'Exp2', style='invis')
            exp.edge('Exp2', 'Exp3', style='invis')
            
            # Connect Experts to Sum
            exp.edge('Exp1', 'MoESum', constraint='true')
            exp.edge('Exp2', 'MoESum', constraint='true')
            exp.edge('Exp3', 'MoESum', constraint='true')
            
            # Connect Gating to Sum
            exp.edge('Gating', 'MoESum', style='dashed', label='weights')

        # --- 3C. Fusion ---
        c.node('Fusion', '<<B>Residual Fusion</B><BR/>Δ total = Δ base + (α * Δ moe)>', 
               shape='note', style='filled', fillcolor='#FFCA28')
        
        c.edge('BaseDelta', 'Fusion')
        c.edge('MoESum', 'Fusion', label=' Δ moe')
        c.edge('MetaGate', 'Fusion', label=' α', style='bold', color='#5E35B1')

    # ==================== 4. 輸出與優化 ====================
    with dot.subgraph(name='cluster_out') as c:
        c.attr(label='Output & Loss', color='grey', style='dotted')
        
        c.node('PrevVal', '<<B>Prev Value</B><BR/>(t-1)>', shape='oval', fillcolor='#F5F5F5')
        c.node('FinalPred', '<<B>Final Pred (t)</B><BR/>Prev + Δ total>', style='filled', fillcolor='#69F0AE')
        c.node('Loss', '<<B>Hybrid Loss</B><BR/>Huber + Directional<BR/>+ Aux BCE>', style='filled', fillcolor='#FF5252', fontcolor='white')
        
        c.edge('Fusion', 'FinalPred')
        c.edge('PrevVal', 'FinalPred', label=' Add')
        c.edge('FinalPred', 'Loss')

    # ==================== 跨區塊連結 (Cross-Cluster Edges) ====================
    # 連接分解分量到 Expert
    # ltail/lhead 用於連接 cluster，但在這裡直接連節點比較準確
    
    # 這裡用隱形的線來引導佈局，實際上概念是 Component -> Experts
    dot.edge('Components', 'Exp1', color='#FFB74D', penwidth='2')
    dot.edge('Components', 'Exp2', color='#FFB74D', penwidth='2')
    dot.edge('Components', 'Exp3', color='#FFB74D', penwidth='2')
    
    # Context 來源
    dot.edge('RawInput', 'Context', style='dotted', color='grey')
    dot.edge('Components', 'Context', style='dotted', color='grey')

    # Render
    output_filename = 'modwt_moe_architecture_hd'
    try:
        dot.render(output_filename, view=True, format='png', cleanup=True)
        print(f"✅ Success! HD Diagram saved to: {output_filename}.png")
    except Exception as e:
        print(f"❌ Error rendering graph: {e}")
        print("Tip: Make sure Graphviz is installed on your system (not just the python package).")

if __name__ == '__main__':
    draw_architecture_diagram_hd()