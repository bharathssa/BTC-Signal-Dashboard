import os
import base64

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    return ""

def main():
    print("Generating Static HTML Report...")
    
    # Check for plots
    plots = [
        ("Cumulative Returns", "plot7_cumulative_returns.png"),
        ("Model Evaluation", "plot2_model_evaluation.png"),
        ("Portfolio Backtest", "plot4_backtest.png"),
        ("Confusion Matrices", "plot3_confusion_matrices.png"),
        ("BTC Technicals", "plot1_btc_technicals.png"),
        ("ETH Technicals", "plot6_eth_technicals.png"),
    ]
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AlphaQuest Capital - 2025 Portfolio Analysis</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #0e1117; color: #fafafa; margin: 0; padding: 40px; }
            .container { max-width: 1000px; margin: 0 auto; background-color: #161b22; padding: 40px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); border-top: 4px solid #f39c12; }
            h1 { color: #ffffff; font-size: 28px; margin-bottom: 5px; }
            h2 { color: #f39c12; font-size: 22px; margin-top: 40px; border-bottom: 1px solid #30363d; padding-bottom: 10px; }
            p.subtitle { color: #8b949e; font-size: 16px; margin-top: 0; margin-bottom: 30px; }
            .metric-container { display: flex; justify-content: space-between; margin-bottom: 30px; gap: 15px; }
            .metric-card { background: linear-gradient(135deg, #1e3a5f, #0f2137); padding: 20px; border-radius: 10px; flex: 1; border-left: 4px solid #f39c12; }
            .metric-title { font-size: 13px; color: #aab4be; text-transform: uppercase; font-weight: bold; letter-spacing: 0.5px; }
            .metric-value { font-size: 28px; color: #ffffff; font-weight: 800; margin: 10px 0 5px 0; }
            .metric-delta { font-size: 14px; color: #2ecc71; }
            .metric-delta.negative { color: #e74c3c; }
            img { max-width: 100%; border-radius: 8px; margin-top: 15px; border: 1px solid #30363d; }
            .footer { margin-top: 50px; text-align: center; color: #8b949e; font-size: 14px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🟡 AlphaQuest Capital — Multi-Asset Signal Dashboard</h1>
            <p class="subtitle">FINTECH 717 — Investment Committee Pitch | 2025 Test Period Report</p>
            
            <div class="metric-container">
                <div class="metric-card">
                    <div class="metric-title">Portfolio Return</div>
                    <div class="metric-value">+12.91%</div>
                    <div class="metric-delta">vs BTC B&H -7.71%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Sharpe Ratio</div>
                    <div class="metric-value">0.44</div>
                    <div class="metric-delta">2025 Choppy Market</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Alpha vs BTC</div>
                    <div class="metric-value">+20.62%</div>
                    <div class="metric-delta">Excess Return</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Max Drawdown</div>
                    <div class="metric-value">-19.27%</div>
                    <div class="metric-delta negative">Worst Loss</div>
                </div>
            </div>
            
            <div style="background-color: #1e3a5f; padding: 15px; border-radius: 8px; margin-bottom: 30px;">
                <strong>🏆 Top Performing Models:</strong> Both BTC and ETH signals are powered by the <strong>GradientBoostingClassifier</strong> (selected by highest out-of-sample ROC-AUC).
            </div>

    """
    
    for title, filename in plots:
        b64 = get_base64_image(filename)
        if b64:
            html += f"<h2>{title}</h2>\n"
            html += f"<img src='data:image/png;base64,{b64}' alt='{title}'>\n"
        else:
            html += f"<h2>{title}</h2>\n<p style='color: #e74c3c;'><i>Image not found: {filename}. Please run the pipeline first to generate charts.</i></p>\n"
            
    html += """
            <div class="footer">
                Generated automatically by the AlphaQuest AI Trading Pipeline.
            </div>
        </div>
    </body>
    </html>
    """
    
    output_path = "AlphaQuest_2025_Portfolio_Report.html"
    with open(output_path, "w") as f:
        f.write(html)
        
    print(f"✅ Report successfully generated at: {os.path.abspath(output_path)}")
    print("Double click this file to open it in your web browser and instantly share it with your team!")

if __name__ == "__main__":
    main()
