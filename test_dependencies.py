try:
    import pandas as pd
    import sklearn
    import fastapi
    import gradio
    
    print("✅ Core dependencies hoạt động tốt!")
    print(f"Pandas version: {pd.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")
    
    # Thử import Feast nếu có
    try:
        import feast
        print(f"✅ Feast version: {feast.__version__}")
    except ImportError:
        print("ℹ️  Feast không được cài đặt")
        
except ImportError as e:
    print(f"❌ Lỗi: {e}")