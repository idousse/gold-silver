"""
Gold/Silver Ratio Trading Strategy - Interactive Web App
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Tuple, List, Optional
import os

# Page config
st.set_page_config(
    page_title="Gold/Silver Ratio Strategy",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean minimal design (dark mode compatible)
st.markdown("""
<style>
    .main-header, h1 {
        font-size: 4rem !important;
        font-weight: 800 !important;
        color: #f0f0f0 !important;
        margin-bottom: 0.5rem !important;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #b0b0b0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #2a2a3d 0%, #1e1e2e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        border: 1px solid #3a3a4d;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-value-large {
        font-size: 4rem;
        font-weight: 800;
        color: #ffffff;
    }
    .metric-value-small {
        font-size: 1.3rem;
        font-weight: 600;
        color: #ffffff;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #a0a0a0;
        margin-top: 0.3rem;
    }
    .metric-label-large {
        font-size: 1.1rem;
        color: #b0b0b0;
        margin-top: 0.5rem;
    }
    .metric-label-small {
        font-size: 0.75rem;
        color: #888;
        margin-top: 0.2rem;
    }
    .gold-metric {
        background: linear-gradient(135deg, #4a4020 0%, #3a3010 100%);
        border: 1px solid #d4a017;
    }
    .gold-metric .metric-value {
        color: #ffd700;
    }
    .silver-metric {
        background: linear-gradient(135deg, #3a3a3a 0%, #2a2a2a 100%);
        border: 1px solid #808080;
    }
    .silver-metric .metric-value {
        color: #c0c0c0;
    }
    .ratio-display {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem;
    }
    .action-buy-gold {
        color: #ffd700;
        background: linear-gradient(135deg, #4a4020 0%, #3a3010 100%);
        padding: 1rem 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        border: 2px solid #d4a017;
    }
    .action-buy-silver {
        color: #e0e0e0;
        background: linear-gradient(135deg, #3a3a3a 0%, #2a2a2a 100%);
        padding: 1rem 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        border: 2px solid #808080;
    }
    .action-hold {
        color: #90EE90;
        background: linear-gradient(135deg, #1a3a1a 0%, #0a2a0a 100%);
        padding: 1rem 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        border: 2px solid #4a8a4a;
    }
    
    /* Streamlit metric styling for dark mode */
    [data-testid="stMetric"] {
        background-color: #1e1e2e !important;
        border: 1px solid #3a3a4d !important;
        border-radius: 10px !important;
        padding: 15px !important;
        width: 100% !important;
        height: 140px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
        padding-top: 20px !important;
    }
    [data-testid="stMetric"] > div {
        width: 100% !important;
    }
    [data-testid="stMetric"] label {
        color: #a0a0a0 !important;
        height: 24px !important;
        display: flex !important;
        align-items: center !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        height: 50px !important;
        display: flex !important;
        align-items: center !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #22c55e !important;
        height: 24px !important;
    }
    
    /* Make columns equal width */
    [data-testid="column"] {
        width: 100% !important;
    }
    
    /* Always show sidebar toggle button */
    [data-testid="stSidebar"] > div:first-child {
        position: relative;
    }
    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="stSidebarExpandButton"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# EURO (historical EUR/USD)
# =============================================================================

def _yf_close_series(df: pd.DataFrame) -> pd.Series:
    """Extract Close as a single Series from yfinance output."""
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        c = df['Close']
        return c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
    return df['Close']


def fetch_eurusd_series(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.Series:
    """Daily EURUSD=X (USD per 1 EUR). Index: dates."""
    pad_start = (start_ts - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    end_pad = (end_ts + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    fx = yf.download("EURUSD=X", start=pad_start, end=end_pad, progress=False)
    s = _yf_close_series(fx)
    if len(s) == 0:
        return s
    s = s.sort_index()
    return s[~s.index.duplicated(keep='last')]


def align_eurusd(eurusd: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Reindex EURUSD to metal dates; forward-fill FX holidays."""
    if len(eurusd) == 0:
        return pd.Series(1.0, index=index)
    aligned = eurusd.reindex(index).ffill().bfill()
    aligned = aligned.ffill().bfill()
    return aligned.fillna(1.0)


def format_eur(value: float, decimals: int = 2) -> str:
    if decimals == 0:
        return f"€{value:,.0f}"
    return f"€{value:,.{decimals}f}"


# Yahoo futures quotes are per troy ounce; UI shows metal weights in grams
OZ_TO_G = 31.1035


@st.cache_data(ttl=3600)
def get_spot_eurusd() -> Optional[float]:
    """Current EURUSD (USD per 1 EUR) for live USD metal quotes."""
    try:
        fx = yf.Ticker("EURUSD=X")
        r = fx.info.get('regularMarketPrice', fx.info.get('previousClose'))
        if r is not None and float(r) > 0:
            return float(r)
    except Exception:
        pass
    return None


# =============================================================================
# DATA FETCHING
# =============================================================================

DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'prices.csv')

@st.cache_data(ttl=86400)  # Cache for 24 hours
def load_csv_data() -> pd.DataFrame:
    """Load historical data from CSV file."""
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
        return df
    return None

@st.cache_data(ttl=3600)  # Cache for 1 hour  
def fetch_recent_data(last_date: str) -> pd.DataFrame:
    """Fetch only recent data from yfinance (after last CSV date)."""
    gold = yf.download("GC=F", start=last_date, end=datetime.now().strftime("%Y-%m-%d"), progress=False)
    silver = yf.download("SI=F", start=last_date, end=datetime.now().strftime("%Y-%m-%d"), progress=False)
    
    def get_close(df):
        if isinstance(df.columns, pd.MultiIndex):
            return df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        return df['Close']
    
    if len(gold) == 0 or len(silver) == 0:
        return None
    
    recent = pd.DataFrame()
    recent['gold_price'] = get_close(gold)
    recent['silver_price'] = get_close(silver)
    recent['ratio'] = recent['gold_price'] / recent['silver_price']
    recent = recent.dropna()
    
    return recent

def fetch_data(start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Fetch data from CSV + recent yfinance data.
    
    Gold/silver prices in trading_data and ratio_data are in EUR per troy ounce,
    using historical EURUSD=X per calendar day (USD per 1 EUR).
    """
    
    # Load historical data from CSV
    csv_data = load_csv_data()
    
    if csv_data is not None:
        # Get the last date in CSV
        last_csv_date = csv_data.index[-1]
        
        # Fetch recent data if needed
        recent_data = fetch_recent_data(last_csv_date.strftime("%Y-%m-%d"))
        
        # Combine CSV and recent data
        if recent_data is not None and len(recent_data) > 0:
            # Remove duplicates (keep CSV data for overlapping dates)
            recent_data = recent_data[recent_data.index > last_csv_date]
            all_data = pd.concat([csv_data, recent_data])
        else:
            all_data = csv_data
    else:
        # Fallback to yfinance if no CSV
        gold = yf.download("GC=F", start=start_date, end=end_date, progress=False)
        silver = yf.download("SI=F", start=start_date, end=end_date, progress=False)
        
        def get_close(df):
            if isinstance(df.columns, pd.MultiIndex):
                return df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
            return df['Close']
        
        all_data = pd.DataFrame()
        all_data['gold_price'] = get_close(gold)
        all_data['silver_price'] = get_close(silver)
        all_data['ratio'] = all_data['gold_price'] / all_data['silver_price']
        all_data = all_data.dropna()
    
    # Filter to requested date range
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    all_data = all_data[(all_data.index >= start_ts) & (all_data.index <= end_ts)]
    
    if len(all_data) == 0:
        empty = pd.DataFrame()
        return empty, empty, pd.Series(dtype=float)
    
    eurusd_hist = fetch_eurusd_series(all_data.index.min(), all_data.index.max())
    eurusd_aligned = align_eurusd(eurusd_hist, all_data.index)
    
    # USD/oz -> EUR/oz: divide by EURUSD (USD per 1 EUR)
    gold_eur = all_data['gold_price'] / eurusd_aligned
    silver_eur = all_data['silver_price'] / eurusd_aligned
    
    ratio_data = pd.DataFrame({
        'gold_price': gold_eur,
        'silver_price': silver_eur,
        'ratio': all_data['ratio'],
    }, index=all_data.index)
    
    trading_data = pd.DataFrame({'GLD': gold_eur, 'SLV': silver_eur}, index=all_data.index)
    
    return ratio_data, trading_data, eurusd_aligned


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_live_ratio():
    """Get current live gold/silver ratio."""
    try:
        gold = yf.Ticker("GC=F")
        silver = yf.Ticker("SI=F")
        gold_price = gold.info.get('regularMarketPrice', gold.info.get('previousClose', 0))
        silver_price = silver.info.get('regularMarketPrice', silver.info.get('previousClose', 0))
        if gold_price and silver_price:
            return gold_price / silver_price, gold_price, silver_price
    except:
        pass
    return None, None, None


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

@dataclass
class Trade:
    date: datetime
    action: str
    ratio: float
    gold_price: float
    silver_price: float
    gold_ounces: float
    silver_ounces: float
    portfolio_value: float
    transaction_cost: float


def run_backtest(ratio_data: pd.DataFrame, trading_data: pd.DataFrame,
                 initial_capital: float, monthly_contribution: float,
                 upper_threshold: float, lower_threshold: float,
                 transaction_fee: float,
                 storage_fee_monthly: float = 0.0,
                 additional_investments: List[Tuple] = None,
                 manual_trades: List[Tuple] = None) -> Tuple[pd.DataFrame, List[Trade], float, float]:
    """Run the backtest with given parameters.
    
    Returns:
        Tuple of (portfolio_df, trades, total_contributions, total_storage_fees)
    """
    
    if additional_investments is None:
        additional_investments = []
    if manual_trades is None:
        manual_trades = []
    
    # Convert additional investment dates to set for quick lookup
    additional_inv_dict = {}
    for inv_date, inv_amount in additional_investments:
        inv_date_ts = pd.Timestamp(inv_date)
        additional_inv_dict[inv_date_ts] = additional_inv_dict.get(inv_date_ts, 0) + inv_amount
    
    # Convert manual trades to dict for lookup (date -> gold_pct)
    manual_trades_dict = {}
    for trade_date, gold_pct in manual_trades:
        trade_date_ts = pd.Timestamp(trade_date)
        manual_trades_dict[trade_date_ts] = gold_pct
    
    # Sort manual trade dates
    manual_trade_dates = sorted(manual_trades_dict.keys())
    
    is_manual_mode = len(manual_trades) > 0
    
    gold_shares = 0.0
    silver_shares = 0.0
    cash = initial_capital
    trades = []
    portfolio_history = []
    total_contributions = initial_capital
    total_storage_fees = 0.0
    
    current_allocation = None
    current_gold_pct = None
    last_contribution_month = None
    last_storage_fee_month = None
    processed_additional_invs = set()
    processed_manual_trades = set()
    
    for i, date in enumerate(ratio_data.index):
        ratio = ratio_data.loc[date, 'ratio']
        gold_price = trading_data.loc[date, 'GLD']
        silver_price = trading_data.loc[date, 'SLV']
        
        # Process additional investments on or after their date
        for inv_date_ts, inv_amount in additional_inv_dict.items():
            if inv_date_ts not in processed_additional_invs and date >= inv_date_ts:
                if current_allocation is not None:
                    fee = inv_amount * transaction_fee
                    invest_amount = inv_amount - fee
                    
                    if current_allocation == 'gold':
                        gold_shares += invest_amount / gold_price
                    elif current_allocation == 'silver':
                        silver_shares += invest_amount / silver_price
                    elif current_gold_pct is not None:
                        gold_shares += (invest_amount * current_gold_pct / 100) / gold_price
                        silver_shares += (invest_amount * (100 - current_gold_pct) / 100) / silver_price
                    else:
                        gold_shares += (invest_amount / 2) / gold_price
                        silver_shares += (invest_amount / 2) / silver_price
                    
                    total_contributions += inv_amount
                    processed_additional_invs.add(inv_date_ts)
        
        # DCA monthly contribution
        current_month = (date.year, date.month)
        if monthly_contribution > 0 and current_allocation is not None:
            if last_contribution_month is None or current_month != last_contribution_month:
                contribution = monthly_contribution
                fee = contribution * transaction_fee
                invest_amount = contribution - fee
                
                if current_allocation == 'gold':
                    gold_shares += invest_amount / gold_price
                elif current_allocation == 'silver':
                    silver_shares += invest_amount / silver_price
                elif current_gold_pct is not None:
                    gold_shares += (invest_amount * current_gold_pct / 100) / gold_price
                    silver_shares += (invest_amount * (100 - current_gold_pct) / 100) / silver_price
                else:
                    gold_shares += (invest_amount / 2) / gold_price
                    silver_shares += (invest_amount / 2) / silver_price
                
                total_contributions += contribution
                last_contribution_month = current_month
        
        # Apply monthly storage fee (deduct from holdings)
        if storage_fee_monthly > 0 and (gold_shares > 0 or silver_shares > 0):
            if last_storage_fee_month is None or current_month != last_storage_fee_month:
                # Calculate storage fee based on current value
                storage_fee_value = (gold_shares * gold_price + silver_shares * silver_price) * storage_fee_monthly
                total_storage_fees += storage_fee_value
                
                # Deduct proportionally from holdings (reduce ounces)
                gold_shares *= (1 - storage_fee_monthly)
                silver_shares *= (1 - storage_fee_monthly)
                last_storage_fee_month = current_month
        
        portfolio_value = gold_shares * gold_price + silver_shares * silver_price + cash
        
        # Determine target allocation
        target_allocation = None
        target_gold_pct = None
        
        if is_manual_mode:
            # Check if there's a manual trade on or before this date
            for trade_date_ts in manual_trade_dates:
                if trade_date_ts not in processed_manual_trades and date >= trade_date_ts:
                    gold_pct = manual_trades_dict[trade_date_ts]
                    if gold_pct == 100:
                        target_allocation = 'gold'
                    elif gold_pct == 0:
                        target_allocation = 'silver'
                    else:
                        target_allocation = 'custom'
                        target_gold_pct = gold_pct
                    processed_manual_trades.add(trade_date_ts)
        else:
            # Automatic mode
            if current_allocation is None:
                if ratio >= upper_threshold:
                    target_allocation = 'silver'
                elif ratio <= lower_threshold:
                    target_allocation = 'gold'
                else:
                    target_allocation = 'both'
            else:
                if ratio >= upper_threshold:
                    target_allocation = 'silver'
                elif ratio <= lower_threshold:
                    target_allocation = 'gold'
        
        # Execute trades if allocation changes
        if target_allocation is not None and (target_allocation != current_allocation or target_gold_pct != current_gold_pct):
            sell_value = gold_shares * gold_price + silver_shares * silver_price
            tx_cost = sell_value * transaction_fee if current_allocation is not None else 0
            available_cash = cash + sell_value - tx_cost
            
            if target_allocation == 'gold':
                buy_cost = available_cash * transaction_fee
                buy_amount = available_cash - buy_cost
                new_gold_shares = buy_amount / gold_price
                new_silver_shares = 0.0
                action = 'BUY_GOLD' if current_allocation is None else 'SWITCH_TO_GOLD'
            elif target_allocation == 'silver':
                buy_cost = available_cash * transaction_fee
                buy_amount = available_cash - buy_cost
                new_gold_shares = 0.0
                new_silver_shares = buy_amount / silver_price
                action = 'BUY_SILVER' if current_allocation is None else 'SWITCH_TO_SILVER'
            elif target_allocation == 'custom':
                buy_cost = available_cash * transaction_fee
                buy_amount = available_cash - buy_cost
                new_gold_shares = (buy_amount * target_gold_pct / 100) / gold_price
                new_silver_shares = (buy_amount * (100 - target_gold_pct) / 100) / silver_price
                action = f'BUY_{target_gold_pct}%_GOLD' if current_allocation is None else f'SWITCH_TO_{target_gold_pct}%_GOLD'
            else:
                buy_cost = available_cash * transaction_fee
                buy_amount = available_cash - buy_cost
                new_gold_shares = (buy_amount / 2) / gold_price
                new_silver_shares = (buy_amount / 2) / silver_price
                action = 'BUY_BOTH'
            
            total_tx_cost = tx_cost + buy_cost
            
            gold_shares = new_gold_shares
            silver_shares = new_silver_shares
            cash = 0.0
            
            trades.append(Trade(
                date=date,
                action=action,
                ratio=ratio,
                gold_price=gold_price,
                silver_price=silver_price,
                gold_ounces=new_gold_shares,
                silver_ounces=new_silver_shares,
                portfolio_value=gold_shares * gold_price + silver_shares * silver_price,
                transaction_cost=total_tx_cost
            ))
            
            current_allocation = target_allocation
            current_gold_pct = target_gold_pct
        
        portfolio_history.append({
            'date': date,
            'ratio': ratio,
            'gold_price': gold_price,
            'silver_price': silver_price,
            'gold_ounces': gold_shares,
            'silver_ounces': silver_shares,
            'portfolio_value': gold_shares * gold_price + silver_shares * silver_price + cash,
            'allocation': current_allocation
        })
    
    portfolio_df = pd.DataFrame(portfolio_history)
    portfolio_df.set_index('date', inplace=True)
    
    return portfolio_df, trades, total_contributions, total_storage_fees


def calculate_metrics(portfolio_df: pd.DataFrame, total_contributions: float,
                      trading_data: pd.DataFrame, monthly_contribution: float,
                      initial_capital: float) -> dict:
    """Calculate performance metrics."""
    final_value = portfolio_df['portfolio_value'].iloc[-1]
    profit = final_value - total_contributions
    total_return = (final_value / total_contributions - 1) * 100
    
    trading_days = len(portfolio_df)
    years = trading_days / 252
    
    # IRR for annualized return
    if monthly_contribution > 0:
        try:
            import numpy_financial as npf
            months = int(years * 12)
            cash_flows = [-initial_capital] + [-monthly_contribution] * months + [final_value]
            monthly_irr = npf.irr(cash_flows)
            annualized_return = ((1 + monthly_irr) ** 12 - 1) * 100
        except:
            annualized_return = ((final_value / total_contributions) ** (1 / years) - 1) * 100
    else:
        annualized_return = ((final_value / total_contributions) ** (1 / years) - 1) * 100
    
    # Max drawdown
    cumulative_max = portfolio_df['portfolio_value'].cummax()
    drawdown = (portfolio_df['portfolio_value'] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min() * 100
    
    # Benchmarks
    initial_gold = trading_data['GLD'].iloc[0]
    final_gold = trading_data['GLD'].iloc[-1]
    gold_return = (final_gold / initial_gold - 1) * 100
    
    initial_silver = trading_data['SLV'].iloc[0]
    final_silver = trading_data['SLV'].iloc[-1]
    silver_return = (final_silver / initial_silver - 1) * 100
    
    return {
        'total_contributions': total_contributions,
        'final_value': final_value,
        'profit': profit,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'years': years,
        'gold_return': gold_return,
        'silver_return': silver_return,
    }


# =============================================================================
# PLOTTING
# =============================================================================

def create_ratio_chart(ratio_data: pd.DataFrame, portfolio_df: pd.DataFrame, 
                       trades: List[Trade], upper_threshold: float, 
                       lower_threshold: float) -> go.Figure:
    """Create the gold/silver ratio chart with colored zones."""
    
    fig = go.Figure()
    
    # Add colored background zones based on allocation
    allocations = portfolio_df['allocation']
    dates = portfolio_df.index
    
    # Find allocation change points
    changes = []
    prev_alloc = None
    for i, (date, alloc) in enumerate(zip(dates, allocations)):
        if alloc != prev_alloc:
            changes.append((date, alloc))
            prev_alloc = alloc
    
    # Add end point
    changes.append((dates[-1], allocations.iloc[-1]))
    
    # Draw colored zones
    for i in range(len(changes) - 1):
        start_date, alloc = changes[i]
        end_date, _ = changes[i + 1]
        
        if alloc == 'gold':
            color = 'rgba(255, 215, 0, 0.2)'  # Gold
        elif alloc == 'silver':
            color = 'rgba(192, 192, 192, 0.3)'  # Silver
        else:
            color = 'rgba(144, 238, 144, 0.2)'  # Green for both
        
        fig.add_vrect(
            x0=start_date, x1=end_date,
            fillcolor=color, layer="below", line_width=0
        )
    
    # Ratio line
    fig.add_trace(go.Scatter(
        x=ratio_data.index,
        y=ratio_data['ratio'],
        mode='lines',
        name='Gold/Silver Ratio',
        line=dict(color='#6366f1', width=2)
    ))
    
    # Threshold lines (only in automatic mode)
    if upper_threshold is not None:
        fig.add_hline(y=upper_threshold, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"Buy Silver ≥{upper_threshold}")
    if lower_threshold is not None:
        fig.add_hline(y=lower_threshold, line_dash="dash", line_color="#22c55e",
                      annotation_text=f"Buy Gold ≤{lower_threshold}")
    
    # Switch points
    for trade in trades:
        if 'GOLD' in trade.action:
            marker_color = '#d4a017'
            symbol = 'triangle-up'
        else:
            marker_color = '#808080'
            symbol = 'triangle-down'
        
        fig.add_trace(go.Scatter(
            x=[trade.date],
            y=[trade.ratio],
            mode='markers',
            marker=dict(size=14, color=marker_color, symbol=symbol, line=dict(width=2, color='white')),
            name=trade.action.replace('_', ' '),
            hovertemplate=f"<b>{trade.action.replace('_', ' ')}</b><br>Date: %{{x}}<br>Ratio: {trade.ratio:.2f}<extra></extra>"
        ))
    
    fig.update_layout(
        title="Gold/Silver Ratio Over Time",
        xaxis_title="Date",
        yaxis_title="Ratio",
        height=500,
        template="plotly_white",
        hovermode='x unified',
        showlegend=False
    )
    
    return fig


def create_holdings_grams_chart(portfolio_df: pd.DataFrame, trades: List[Trade]) -> go.Figure:
    """Create chart showing gold/silver weight over time (grams)."""
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Gold (g)", "Silver (g)"),
                        vertical_spacing=0.1)
    
    fig.add_trace(go.Scatter(
        x=portfolio_df.index,
        y=portfolio_df['gold_ounces'] * OZ_TO_G,
        fill='tozeroy',
        fillcolor='rgba(255, 215, 0, 0.3)',
        line=dict(color='#d4a017', width=2),
        name='Gold (g)'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=portfolio_df.index,
        y=portfolio_df['silver_ounces'] * OZ_TO_G,
        fill='tozeroy',
        fillcolor='rgba(192, 192, 192, 0.4)',
        line=dict(color='#808080', width=2),
        name='Silver (g)'
    ), row=2, col=1)
    
    for trade in trades:
        gold_g = trade.gold_ounces * OZ_TO_G
        silver_g = trade.silver_ounces * OZ_TO_G
        
        fig.add_trace(go.Scatter(
            x=[trade.date], y=[gold_g],
            mode='markers', marker=dict(size=10, color='#d4a017', symbol='diamond'),
            showlegend=False,
            hovertemplate=f"<b>{trade.action}</b><br>Gold: {gold_g:.2f} g<extra></extra>"
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=[trade.date], y=[silver_g],
            mode='markers', marker=dict(size=10, color='#808080', symbol='diamond'),
            showlegend=False,
            hovertemplate=f"<b>{trade.action}</b><br>Silver: {silver_g:.2f} g<extra></extra>"
        ), row=2, col=1)
    
    fig.update_layout(
        height=500,
        template="plotly_white",
        showlegend=False
    )
    
    fig.update_yaxes(title_text="g", row=1, col=1)
    fig.update_yaxes(title_text="g", row=2, col=1)
    
    return fig


def create_portfolio_chart(portfolio_df: pd.DataFrame, trading_data: pd.DataFrame,
                           initial_capital: float) -> go.Figure:
    """Create portfolio value chart vs benchmarks (amounts in EUR)."""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=portfolio_df.index,
        y=portfolio_df['portfolio_value'],
        mode='lines',
        name='Strategy',
        line=dict(color='#6366f1', width=3)
    ))
    
    gold_normalized = (trading_data['GLD'] / trading_data['GLD'].iloc[0]) * initial_capital
    fig.add_trace(go.Scatter(
        x=trading_data.index,
        y=gold_normalized,
        mode='lines',
        name='Gold Buy & Hold',
        line=dict(color='#d4a017', width=2, dash='dot')
    ))
    
    silver_normalized = (trading_data['SLV'] / trading_data['SLV'].iloc[0]) * initial_capital
    fig.add_trace(go.Scatter(
        x=trading_data.index,
        y=silver_normalized,
        mode='lines',
        name='Silver Buy & Hold',
        line=dict(color='#808080', width=2, dash='dot')
    ))
    
    fig.update_layout(
        title="Portfolio Value: Switching vs Buy & Hold",
        xaxis_title="Date",
        yaxis_title="Value (€)",
        height=400,
        template="plotly_white",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown('<h1 style="font-size: 4rem !important; font-weight: 800; color: #f0f0f0; margin-bottom: 0.5rem; text-align: center;">🥇 Gold/Silver Ratio Strategy Simulator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">A simple pairs trading strategy: Buy silver when the ratio is high, buy gold when the ratio is low.</p>', unsafe_allow_html=True)
    
    # Sidebar - Parameters
    st.sidebar.header("⚙️ Parameters")
    
    st.sidebar.subheader("📅 Date Range")
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start", datetime(2000, 8, 30), 
                                  min_value=datetime(2000, 8, 30),
                                  max_value=datetime.now())
    end_date = col2.date_input("End", datetime.now(),
                                min_value=datetime(2000, 8, 30),
                                max_value=datetime.now())
    
    # Clear manual trades if date range changes
    if 'manual_trades' not in st.session_state:
        st.session_state.manual_trades = []
    if 'prev_start_date' not in st.session_state:
        st.session_state.prev_start_date = start_date
    if 'prev_end_date' not in st.session_state:
        st.session_state.prev_end_date = end_date
    
    if start_date != st.session_state.prev_start_date or end_date != st.session_state.prev_end_date:
        st.session_state.manual_trades = []
        st.session_state.prev_start_date = start_date
        st.session_state.prev_end_date = end_date
    
    st.sidebar.subheader("💰 Investment (EUR)")
    initial_capital = st.sidebar.number_input("Initial Investment (€)", min_value=100, value=10000, step=1000)
    monthly_contribution = st.sidebar.number_input("Monthly Contribution (€)", min_value=0, value=0, step=50)
    
    # Additional investments
    st.sidebar.subheader("💵 Additional Investments")
    
    # Initialize session state for additional investments
    if 'additional_investments' not in st.session_state:
        st.session_state.additional_investments = []
    
    # Display existing additional investments
    investments_to_remove = []
    for i, inv in enumerate(st.session_state.additional_investments):
        col1, col2, col3 = st.sidebar.columns([2, 2, 1])
        with col1:
            new_date = col1.date_input(
                f"Date {i+1}", 
                value=inv['date'],
                min_value=start_date,
                max_value=end_date,
                key=f"inv_date_{i}"
            )
            st.session_state.additional_investments[i]['date'] = new_date
        with col2:
            new_amount = col2.number_input(
                "€",
                min_value=0,
                value=inv['amount'],
                step=1000,
                key=f"inv_amount_{i}"
            )
            st.session_state.additional_investments[i]['amount'] = new_amount
        with col3:
            if col3.button("❌", key=f"remove_{i}"):
                investments_to_remove.append(i)
    
    # Remove marked investments
    for i in sorted(investments_to_remove, reverse=True):
        st.session_state.additional_investments.pop(i)
    
    # Add new investment button
    if st.sidebar.button("➕ Add Investment"):
        st.session_state.additional_investments.append({
            'date': start_date + timedelta(days=365),
            'amount': 5000
        })
        st.rerun()
    
    additional_investments = [(inv['date'], inv['amount']) for inv in st.session_state.additional_investments]
    
    st.sidebar.subheader("📊 Strategy Mode")
    strategy_mode = st.sidebar.radio("Mode", ["Automatic", "Manual"], horizontal=True)
    
    if strategy_mode == "Automatic":
        upper_threshold = st.sidebar.slider("Buy Silver when ratio ≥", 70, 100, 85)
        lower_threshold = st.sidebar.slider("Buy Gold when ratio ≤", 50, 80, 75)
        manual_trades = []
        
        if lower_threshold >= upper_threshold:
            st.sidebar.error("Lower threshold must be less than upper threshold!")
            return
    else:
        upper_threshold = None
        lower_threshold = None
        
        # Initialize session state for manual trades
        if 'manual_trades' not in st.session_state:
            st.session_state.manual_trades = []
        
        st.sidebar.markdown("**Define your trades:**")
        
        # Display existing manual trades
        trades_to_remove = []
        for i, trade in enumerate(st.session_state.manual_trades):
            # Trade header with delete button on same line
            header_col, delete_col = st.sidebar.columns([3, 1])
            with header_col:
                st.markdown(f"**Trade {i+1}**")
            with delete_col:
                if st.button("❌", key=f"remove_trade_{i}"):
                    trades_to_remove.append(i)
            
            new_date = st.sidebar.date_input(
                f"Date",
                value=trade['date'],
                min_value=start_date,
                max_value=end_date,
                key=f"trade_date_{i}"
            )
            st.session_state.manual_trades[i]['date'] = new_date
            
            allocation_type = st.sidebar.selectbox(
                "Allocation",
                ["100% Gold", "100% Silver", "Custom Ratio"],
                index=["100% Gold", "100% Silver", "Custom Ratio"].index(trade.get('allocation_type', '100% Gold')),
                key=f"trade_alloc_{i}"
            )
            st.session_state.manual_trades[i]['allocation_type'] = allocation_type
            
            if allocation_type == "Custom Ratio":
                gold_pct = st.sidebar.slider(
                    "Gold %",
                    0, 100, 
                    trade.get('gold_pct', 50),
                    key=f"trade_gold_pct_{i}"
                )
                st.session_state.manual_trades[i]['gold_pct'] = gold_pct
            else:
                st.session_state.manual_trades[i]['gold_pct'] = 100 if allocation_type == "100% Gold" else 0
            
            st.sidebar.markdown("---")
        
        # Remove marked trades
        for i in sorted(trades_to_remove, reverse=True):
            st.session_state.manual_trades.pop(i)
        
        # Add new trade button
        if st.sidebar.button("➕ Add Trade"):
            st.session_state.manual_trades.append({
                'date': start_date,
                'allocation_type': '100% Gold',
                'gold_pct': 100
            })
            st.rerun()
        
        # Convert to list for backtest
        manual_trades = [(trade['date'], trade['gold_pct']) for trade in st.session_state.manual_trades]
        
        if len(manual_trades) == 0:
            st.sidebar.warning("Add at least one trade to start!")
            st.info("👈 Switch to **Manual** mode and add at least one trade to see results.")
            return
    
    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date!")
        return
    
    st.sidebar.subheader("💸 Fees")
    transaction_fee = st.sidebar.number_input("Transaction Fee (%)", min_value=0.0, max_value=10.0, value=0.3, step=0.1, format="%.2f") / 100
    storage_fee_monthly = st.sidebar.number_input("Storage Fee (%/month)", min_value=0.0, max_value=5.0, value=0.0, step=0.01, format="%.2f", help="Monthly fee charged for storing physical metals (e.g., 0.05% = 0.6% annually)") / 100
    
    # Fetch and process data first (needed for fallback)
    with st.spinner("Fetching market data..."):
        ratio_data, trading_data, eurusd_series = fetch_data(str(start_date), str(end_date))
    
    if len(ratio_data) == 0:
        st.error("No data available for the selected date range.")
        return
    
    # Live ratio display (with fallback to most recent historical data)
    live_ratio, gold_price_usd, silver_price_usd = get_live_ratio()
    last_hist_date = ratio_data.index[-1].strftime("%Y-%m-%d")
    spot_eurusd = get_spot_eurusd()
    if spot_eurusd is None or spot_eurusd <= 0:
        spot_eurusd = float(eurusd_series.iloc[-1]) if len(eurusd_series) else 1.0
    
    is_live = (
        live_ratio is not None
        and gold_price_usd is not None
        and silver_price_usd is not None
        and gold_price_usd > 0
        and silver_price_usd > 0
    )
    if is_live:
        gold_price_display = (gold_price_usd / spot_eurusd) / OZ_TO_G
        silver_price_display = (silver_price_usd / spot_eurusd) / OZ_TO_G
        ratio_label = "Current Gold/Silver Ratio"
    else:
        live_ratio = ratio_data['ratio'].iloc[-1]
        gold_price_display = ratio_data['gold_price'].iloc[-1] / OZ_TO_G
        silver_price_display = ratio_data['silver_price'].iloc[-1] / OZ_TO_G
        ratio_label = f"Latest Gold/Silver Ratio ({last_hist_date})"
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="padding: 2rem;">
            <div class="metric-value-large">{live_ratio:.2f}</div>
            <div class="metric-label-large">{ratio_label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card gold-metric" style="padding: 0.8rem; margin-bottom: 0.5rem;">
            <div class="metric-value-small">{format_eur(gold_price_display)}</div>
            <div class="metric-label-small">Gold (per g)</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card silver-metric" style="padding: 0.8rem;">
            <div class="metric-value-small">{format_eur(silver_price_display)}</div>
            <div class="metric-label-small">Silver (per g)</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show warning if using historical data
    if not is_live:
        st.caption("⚠️ Live prices unavailable. Showing most recent historical data (EUR using that day's EUR/USD).")
    
    # Current action recommendation (only in automatic mode)
    if upper_threshold is not None and lower_threshold is not None:
        st.markdown("---")
        signal_col, disclaimer_col = st.columns([2, 1])
        
        with signal_col:
            if live_ratio >= upper_threshold:
                st.markdown(f'<div class="action-buy-silver">📍 Current Signal: BUY SILVER (Ratio {live_ratio:.2f} ≥ {upper_threshold})</div>', unsafe_allow_html=True)
            elif live_ratio <= lower_threshold:
                st.markdown(f'<div class="action-buy-gold">📍 Current Signal: BUY GOLD (Ratio {live_ratio:.2f} ≤ {lower_threshold})</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="action-hold">📍 Current Signal: HOLD (Ratio {live_ratio:.2f} between {lower_threshold}-{upper_threshold})</div>', unsafe_allow_html=True)
        
        with disclaimer_col:
            st.markdown("""
            <div style="color: #c0c0c0; font-size: 1rem; font-style: italic; padding: 0.5rem; text-align: center;">
                ⚠️ This is what I would do based on this strategy.<br>
                <strong>Not financial advice.</strong> For educational purposes only.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("---")
        st.markdown('<div class="action-hold">📍 Manual Mode - No automatic signal</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Run backtest
    portfolio_df, trades, total_contributions, total_storage_fees = run_backtest(
        ratio_data, trading_data,
        initial_capital, monthly_contribution,
        upper_threshold, lower_threshold,
        transaction_fee,
        storage_fee_monthly,
        additional_investments,
        manual_trades
    )
    
    # Calculate metrics
    metrics = calculate_metrics(portfolio_df, total_contributions, trading_data,
                                monthly_contribution, initial_capital)
    
    # Performance Summary
    st.subheader("📈 Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Invested", format_eur(metrics['total_contributions'], 0))
    with col2:
        st.metric("Final Value", format_eur(metrics['final_value'], 0))
    with col3:
        st.metric("Profit", format_eur(metrics['profit'], 0), 
                  delta=f"{metrics['total_return']:.1f}%")
    with col4:
        st.metric("Annual Return", f"{metrics['annualized_return']:.2f}%")
    
    # Strategy vs Benchmarks
    st.subheader("🏆 Switching vs Buying and Holding")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("With Switching", f"{metrics['total_return']:.1f}%")
    with col2:
        gold_diff = metrics['total_return'] - metrics['gold_return']
        st.metric("If you buy Gold and hold", f"{metrics['gold_return']:.1f}%",
                  delta=f"Switching beats by {gold_diff:.0f}%", delta_color="off")
    with col3:
        silver_diff = metrics['total_return'] - metrics['silver_return']
        st.metric("If you buy Silver and hold", f"{metrics['silver_return']:.1f}%",
                  delta=f"Switching beats by {silver_diff:.0f}%", delta_color="off")
    
    st.markdown("---")
    
    # Charts
    st.subheader("📊 Gold/Silver Ratio")
    ratio_chart = create_ratio_chart(ratio_data, portfolio_df, trades, upper_threshold, lower_threshold)
    st.plotly_chart(ratio_chart, use_container_width=True)
    
    st.subheader("⚖️ Metal Holdings (g)")
    holdings_chart = create_holdings_grams_chart(portfolio_df, trades)
    st.plotly_chart(holdings_chart, use_container_width=True)
    
    st.subheader("💹 Portfolio Value")
    portfolio_chart = create_portfolio_chart(portfolio_df, trading_data, initial_capital)
    st.plotly_chart(portfolio_chart, use_container_width=True)
    
    # Trade History
    st.subheader("📋 Trade History")
    
    if trades:
        trade_data = []
        
        for i, trade in enumerate(trades):
            gold_weight = trade.gold_ounces * OZ_TO_G
            silver_weight = trade.silver_ounces * OZ_TO_G
            
            if trade.gold_ounces > 0 and trade.silver_ounces == 0:
                price_per_g = trade.gold_price / OZ_TO_G
                price_str = f"€{price_per_g:.2f}/g"
            elif trade.silver_ounces > 0 and trade.gold_ounces == 0:
                price_per_g = trade.silver_price / OZ_TO_G
                price_str = f"€{price_per_g:.2f}/g"
            else:
                gold_ppg = trade.gold_price / OZ_TO_G
                silver_ppg = trade.silver_price / OZ_TO_G
                price_str = f"Au:€{gold_ppg:.2f} Ag:€{silver_ppg:.2f}/g"
            
            trade_data.append({
                "Date": trade.date.strftime("%Y-%m-%d"),
                "Action": trade.action.replace("_", " "),
                "Ratio": f"{trade.ratio:.2f}",
                "Price/g": price_str,
                "Gold (g)": f"{gold_weight:.2f}" if trade.gold_ounces > 0 else "-",
                "Silver (g)": f"{silver_weight:.2f}" if trade.silver_ounces > 0 else "-",
                "Portfolio Value": format_eur(trade.portfolio_value),
                "Fee": format_eur(trade.transaction_cost)
            })
        
        st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)
        
        total_tx_fees = sum(t.transaction_cost for t in trades)
        storage_fees_display = f" | **Storage Fees:** {format_eur(total_storage_fees)}" if total_storage_fees > 0 else ""
        st.info(f"**Total Trades:** {len(trades)} | **Transaction Fees:** {format_eur(total_tx_fees)}{storage_fees_display}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9rem;">
        <p>Data: Yahoo Finance (GC=F, SI=F futures). Amounts in EUR; futures quoted per troy oz, converted with <strong>daily</strong> EUR/USD (EURUSD=X). Metal weights and spot cards shown in <strong>grams</strong> (1 oz t = 31.1035 g).</p>
        <p>⚠️ For educational purposes only. Past performance does not guarantee future results.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
