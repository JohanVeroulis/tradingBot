class OrderExecutor:
    def __init__(self, session, symbol):
        self.session = session
        self.symbol = symbol

    def place_order(self, signal, qty=1):
        if signal == 'BUY':
            print("Placing a BUY order...")
            order_response = self.session.place_active_order(
                symbol=self.symbol,
                side="Buy",
                order_type="Market",
                qty=qty,
                time_in_force="GoodTillCancel"
            )
            print("Buy Order Response:", order_response)
        elif signal == 'SELL':
            print("Placing a SELL order...")
            order_response = self.session.place_active_order(
                symbol=self.symbol,
                side="Sell",
                order_type="Market",
                qty=qty,
                time_in_force="GoodTillCancel"
            )
            print("Sell Order Response:", order_response)
        else:
            print("Signal is HOLD. No order placed.")
