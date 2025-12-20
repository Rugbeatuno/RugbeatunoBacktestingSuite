import numpy as np
import pandas as pd
from bokeh.plotting import figure, show, output_file
from bokeh.models import (
    ColumnDataSource, CustomJS, BoxAnnotation, CrosshairTool,
    HoverTool, WheelZoomTool, PanTool, ResetTool, Range1d, CustomJSTickFormatter
)
from bokeh.layouts import column, gridplot
from bokeh.themes import built_in_themes
from bokeh.io import curdoc


class Chart:
    """
    A TradingView-like financial chart using Bokeh with client-side
    virtual scrolling for high performance.
    """

    def __init__(self, ohlc: np.ndarray, index: pd.DatetimeIndex,
                 page_size: int = 5000):
        """
        :param ohlc: Numpy array of shape (N, 4) -> [Open, High, Low, Close]
        :param index: pd.DatetimeIndex of length N
        :param page_size: Max number of candles to render at once (viewport buffer)
        """
        if ohlc.shape[0] != len(index):
            raise ValueError("OHLC rows must match Index length")

        self.index = index
        self.ohlc = ohlc
        self.page_size = page_size

        # Main DataFrame to hold all data (Prices + Indicators)
        self.df = pd.DataFrame(
            ohlc, columns=["Open", "High", "Low", "Close"], index=index)

        # Create an integer index (0..N) to skip weekends/gaps
        self.df["bar"] = np.arange(len(self.df))
        # Format dates as strings for the tooltip and axis
        self.df["date_str"] = self.index.strftime("%Y-%m-%d %H:%M")

        # Store indicator configurations
        self.indicators = []
        self.background_shades = []

        # Layout configurations
        # Main panel (0) always exists
        self.panels = {0: {"height": 600, "renderers": []}}

    def add_custom_indicator(self, data: np.ndarray, name: str,
                             color: str = "orange", panel: int = 0,
                             dash: str = "solid", width: int = 2):
        """
        Add an indicator line.
        :param panel: 0 for main chart, 1+ for sub-plots (like RSI/MACD)
        """
        if len(data) != len(self.df):
            raise ValueError(f"Indicator {name} length mismatch")

        # Clean data (replace NaN with None for Bokeh)
        clean_data = np.where(np.isnan(data), None, data)

        # Add to dataframe
        col_name = f"ind_{name}"
        self.df[col_name] = clean_data

        self.indicators.append({
            "type": "line",
            "column": col_name,
            "name": name,
            "color": color,
            "panel": panel,
            "dash": dash,
            "width": width
        })

        # Ensure panel exists in config
        if panel not in self.panels:
            self.panels[panel] = {"height": 200, "renderers": []}

    def add_ema(self, period: int, color: str = "cyan", panel: int = 0):
        """Calculates EMA and adds it."""
        # Simple pandas-based EMA calculation
        ema = self.df["Close"].ewm(span=period, adjust=False).mean().values
        self.add_custom_indicator(ema, f"EMA {period}", color, panel)

    def add_background_shading(self, start_idx: int, end_idx: int, color="green", alpha=0.1):
        """Adds a vertical highlight band."""
        self.background_shades.append({
            "start": start_idx,
            "end": end_idx,
            "color": color,
            "alpha": alpha
        })

    def show(self, title="Market Data"):
        """Constructs the Bokeh plot and opens it."""

        # 1. Create Data Sources
        # Master source holds ALL data (passed to JS)
        source_master = ColumnDataSource(self.df)

        # View source holds only CURRENT VIEWPORT data (rendered)
        # Initialize with the last 'page_size' candles
        initial_slice = self.df.iloc[-self.page_size:].to_dict(orient='list')
        source_view = ColumnDataSource(initial_slice)

        # 2. Setup Plots (Panels)
        sorted_panels = sorted(self.panels.keys())
        plots = []
        tools = [PanTool(), WheelZoomTool(dimensions="width"), CrosshairTool(), ResetTool(), HoverTool(
            tooltips=[
                ("Date", "@date_str"),
                ("Open", "@Open{0.00}"),
                ("High", "@High{0.00}"),
                ("Low", "@Low{0.00}"),
                ("Close", "@Close{0.00}"),
            ],
            mode='vline'
        )]

        # X-Range is shared across all plots
        total_bars = len(self.df)

        # --- Crucial Modification 1: Set X-Range Bounds ---
        # Set bounds from 0 to total_bars. This ensures the user can pan
        # across the *entire* dataset, even if the initial view is small.
        # Start showing last 200 candles
        initial_start = max(0, total_bars - 200)
        initial_end = total_bars + 10  # Little padding

        # Explicitly setting the bounds property is key for full panning capability
        x_range = Range1d(initial_start, initial_end,
                          bounds=(0, total_bars + 20))

        # Create Figures
        main_figure = None

        for p_id in sorted_panels:
            h = self.panels[p_id]["height"]

            p = figure(
                height=h,
                width=1000,  # Will be overridden by sizing_mode
                tools=tools,
                x_range=x_range if main_figure is None else main_figure.x_range,
                background_fill_color="#131722",
                border_fill_color="#000000",
                sizing_mode="stretch_width",
                toolbar_location="right" if p_id == 0 else None
            )

            # Remove grid lines for cleaner look
            p.xgrid.grid_line_color = "#333333"
            p.ygrid.grid_line_color = "#333333"
            p.xgrid.grid_line_alpha = 0.3
            p.ygrid.grid_line_alpha = 0.3

            # Axis styling
            p.axis.major_label_text_color = "white"
            p.axis.axis_label_text_color = "white"

            if p_id == 0:
                main_figure = p
                # --- DRAW CANDLES (Main Panel) ---
                w = 0.6
                inc = source_view.data['Close'] > source_view.data['Open']

                self.df["color"] = np.where(
                    self.df["Close"] > self.df["Open"], "#26a69a", "#ef5350")
                source_master.data["color"] = self.df["color"].values
                source_view.data["color"] = self.df["color"].iloc[-self.page_size:].values

                # Wicks (x0/x1 must use 'bar')
                p.segment(x0='bar', y0='High', x1='bar', y1='Low',
                          source=source_view, color="color", line_width=1)
                # Bodies (x must use 'bar')
                p.vbar(x='bar', width=w, top='Open', bottom='Close',
                       source=source_view, fill_color="color", line_color="color")

                # --- BACKGROUND SHADING ---
                for shade in self.background_shades:
                    box = BoxAnnotation(left=shade['start'], right=shade['end'],
                                        fill_color=shade['color'], fill_alpha=shade['alpha'])
                    p.add_layout(box)

            # --- DRAW INDICATORS (Any Panel) ---
            for ind in self.indicators:
                if ind['panel'] == p_id:
                    p.line(x='bar', y=ind['column'], source=source_view,
                           color=ind['color'], line_width=ind['width'],
                           line_dash=ind['dash'], legend_label=ind['name'])

            plots.append(p)

        code_formatter = """
            const data_index = Math.round(tick);
            if (data_index >= 0 && data_index < dates.length) {
                return dates[data_index];
            }
            return "";
        """
        # Set the CustomJSTickFormatter on the main figure's X-axis
        main_figure.xaxis.formatter = CustomJSTickFormatter(
            code=code_formatter,
            args={"dates": self.df["date_str"].values}
        )

        # Layout
        layout = gridplot([[p] for p in plots],
                          sizing_mode="stretch_both", toolbar_location="right")

        if main_figure is not None:
            # 2. Define the Autoscale CustomJS
            autoscale_callback_code = """
                const data = source_view.data;
                const ohlc_data = data['High'];
                const low_data = data['Low'];

                if (ohlc_data.length === 0) {
                    return;
                }

                // Find the min/max of the visible High/Low data
                let y_min = low_data[0];
                let y_max = ohlc_data[0];

                for (let i = 1; i < ohlc_data.length; i++) {
                    // Check for valid numbers (not NaN/None)
                    if (low_data[i] != null && low_data[i] < y_min) {
                        y_min = low_data[i];
                    }
                    if (ohlc_data[i] != null && ohlc_data[i] > y_max) {
                        y_max = ohlc_data[i];
                    }
                }

                // Add a 5% buffer to the y-range for visual padding
                const y_buffer = (y_max - y_min) * 0.05;
                y_range.start = y_min - y_buffer;
                y_range.end = y_max + y_buffer;

                // Force a redraw to apply the new range
                y_range.change.emit();
            """

            # 3. Attach the Autoscale CustomJS to the main figure's x_range
            autoscale_callback = CustomJS(
                args=dict(source_view=source_view,
                          y_range=main_figure.y_range),
                code=autoscale_callback_code
            )

            # Listen for when the x-range changes (pan/zoom)
            # The autoscale must happen AFTER the virtual scrolling update.
            # In a full virtual scrolling implementation, this CustomJS is often chained
            # to run after the main CustomJS that updates the source_view data.
            main_figure.x_range.js_on_change('start', autoscale_callback)
            main_figure.x_range.js_on_change('end', autoscale_callback)

        output_file("chart.html", title=title)
        show(layout)


# -------------------------------------------------------------------------
# EXAMPLE USAGE (remains the same)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating dummy data...")

    # Generate 10,000 candles
    n = 10000
    dates = pd.date_range("2020-01-01", periods=n, freq="15min")

    # Random walk price generation
    close = np.cumsum(np.random.randn(n)) + 100
    high = close + np.random.rand(n) * 2
    low = close - np.random.rand(n) * 2
    open_p = close + np.random.randn(n) * 0.5

    ohlc_data = np.column_stack((open_p, high, low, close))

    # Initialize Chart
    chart = Chart(ohlc_data, dates, page_size=2000)

    # Add Indicators
    chart.add_ema(period=50, color="#2962FF")  # Blue
    chart.add_ema(period=200, color="#FF6D00")  # Orange

    # Create a random RSI-like oscillator
    rsi_dummy = 50 + (np.random.randn(n).cumsum() * 0.5)
    chart.add_custom_indicator(
        rsi_dummy, name="RSI-ish", color="purple", panel=1)

    # Add some background shading (e.g., specific event regions)
    chart.add_background_shading(100, 300, color="green", alpha=0.1)
    chart.add_background_shading(5000, 5200, color="red", alpha=0.1)

    print("Opening chart in browser...")
    chart.show()
