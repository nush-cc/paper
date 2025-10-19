import datetime
import investpy
import pandas as pd

def Myinvestpy():
    """
    方法1: 使用investpy套件 (推薦)
    """
    print("=== 方法1: 使用investpy套件 ===")

    try:
        # 使用investpy抓取USD/TWD匯率資料
        print("正在從Investing.com抓取USD/TWD匯率資料...")

        # 設定日期範圍
        start_date = "01/01/2000"  # investpy使用DD/MM/YYYY格式
        end_date = datetime.datetime.now().strftime("%d/%m/%Y")


        # 抓取USD/TWD歷史資料
        data = investpy.get_currency_cross_historical_data(
            currency_cross='USD/TWD',
            from_date=start_date,
            to_date=end_date,
        )

        # 處理資料格式
        df = pd.DataFrame({
            'Date': data.index.strftime('%Y-%m-%d'),
            'Close': data['Close']
        })

        # 重設索引
        df = df.reset_index(drop=True)

        print(f"成功抓取 {len(df)} 筆資料")
        print(f"資料範圍: {df['Date'].iloc[0]} 到 {df['Date'].iloc[-1]}")

        # 儲存為CSV
        filename = f"../dataset/USD_TWD.csv"
        df.to_csv(filename, index=False, encoding='utf-8')

        print(f"資料已儲存至: {filename}")

        # 顯示預覽
        print("\n前5筆資料:")
        print(df.head())
        print("\n後5筆資料:")
        print(df.tail())

        return df

    except Exception as e:
        print(f"investpy方法發生錯誤: {str(e)}")
        print("可能原因: investpy套件可能需要更新或Investing.com網站結構改變")
        return None


if __name__ == "__main__":
    Myinvestpy()