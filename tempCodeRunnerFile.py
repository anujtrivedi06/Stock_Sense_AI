predicted_price = test_df['close'].values[-1] * (1 + predictions[-1])
        actual_price = test_df['close'].values[-1] * (1 + test_df['target'].values[-1])