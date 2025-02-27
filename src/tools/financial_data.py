from smolagents import Tool
import yfinance as yf
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta

class FinancialDataTool(Tool):
    name = "financial_data"
    description = (
        "Retrieves financial information and calculates key financial ratios for a given ticker symbol (e.g., NVDA, AAPL) "
        "using Yahoo Finance data. This tool can fetch stock price, financials, and calculate important metrics."
    )
    inputs = {
        "ticker": {
            "type": "string",
            "description": "The stock ticker symbol to analyze (e.g., NVDA, AAPL, MSFT)",
        },
        "data_type": {
            "type": "string",
            "description": "Type of financial data to retrieve: 'overview', 'ratios', 'income', 'balance', 'cash_flow', 'price', or 'all'",
            "default": "overview"
        },
        "period": {
            "type": "string",
            "description": "The time period for historical data: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'",
            "default": "1y"
        }
    }
    output_type = "string"

    def forward(self, ticker: str, data_type: str = "overview", period: str = "1y") -> str:
        ticker = ticker.upper()
        
        try:
            # Get the ticker data from yfinance
            stock = yf.Ticker(ticker)
            
            if data_type == "all":
                # Fetch all data types and combine
                overview = self._get_company_overview(stock)
                ratios = self._calculate_ratios(stock)
                income = self._get_income_statement(stock)
                balance = self._get_balance_sheet(stock)
                cash_flow = self._get_cash_flow(stock)
                price_data = self._get_price_data(stock, period)
                
                return f"""
## Financial Analysis for {ticker}

### Company Overview
{overview}

### Recent Price Information
{price_data}

### Key Financial Ratios
{ratios}

### Income Statement Highlights
{income}

### Balance Sheet Highlights
{balance}

### Cash Flow Highlights
{cash_flow}
"""
            
            elif data_type == "overview":
                return self._get_company_overview(stock)
            elif data_type == "ratios":
                return self._calculate_ratios(stock)
            elif data_type == "income":
                return self._get_income_statement(stock)
            elif data_type == "balance":
                return self._get_balance_sheet(stock)
            elif data_type == "cash_flow":
                return self._get_cash_flow(stock)
            elif data_type == "price":
                return self._get_price_data(stock, period)
            else:
                return f"Invalid data_type: {data_type}. Please use 'overview', 'ratios', 'income', 'balance', 'cash_flow', 'price', or 'all'."
        
        except Exception as e:
            return f"Error retrieving financial data for {ticker}: {str(e)}"

    def _get_company_overview(self, stock: yf.Ticker) -> str:
        try:
            info = stock.info
            
            # Format key company information
            market_cap = info.get('marketCap', 'N/A')
            if market_cap != 'N/A':
                market_cap = self._format_number(market_cap)
                
            pe_ratio = info.get('trailingPE', 'N/A')
            if pe_ratio != 'N/A':
                pe_ratio = f"{pe_ratio:.2f}"
                
            dividend_yield = info.get('dividendYield', 'N/A')
            if dividend_yield != 'N/A':
                dividend_yield = f"{dividend_yield * 100:.2f}%"
                
            fifty_two_week_high = info.get('fiftyTwoWeekHigh', 'N/A')
            if fifty_two_week_high != 'N/A':
                fifty_two_week_high = f"${fifty_two_week_high:.2f}"
                
            fifty_two_week_low = info.get('fiftyTwoWeekLow', 'N/A')
            if fifty_two_week_low != 'N/A':
                fifty_two_week_low = f"${fifty_two_week_low:.2f}"
            
            overview = f"""
**{info.get('shortName', stock.ticker)}** ({stock.ticker})
Sector: {info.get('sector', 'N/A')}
Industry: {info.get('industry', 'N/A')}
Market Cap: ${market_cap}
P/E Ratio: {pe_ratio}
Dividend Yield: {dividend_yield}
52-Week High: {fifty_two_week_high}
52-Week Low: {fifty_two_week_low}

Description:
{info.get('longBusinessSummary', 'No description available.')}
"""
            return overview
        except Exception as e:
            return f"Error formatting company overview: {str(e)}"

    def _get_income_statement(self, stock: yf.Ticker) -> str:
        try:
            # Get the income statement (annual)
            income_stmt = stock.income_stmt
            
            if income_stmt.empty:
                return "Income statement data not available"
            
            # Most recent year is the last column
            latest_year = income_stmt.columns[0]
            year = latest_year.year
            
            # Extract key metrics
            total_revenue = income_stmt.loc['Total Revenue', latest_year] if 'Total Revenue' in income_stmt.index else None
            gross_profit = income_stmt.loc['Gross Profit', latest_year] if 'Gross Profit' in income_stmt.index else None
            operating_income = income_stmt.loc['Operating Income', latest_year] if 'Operating Income' in income_stmt.index else None
            net_income = income_stmt.loc['Net Income', latest_year] if 'Net Income' in income_stmt.index else None
            
            # Format the output
            income_summary = f"""
### Income Statement ({year})
Revenue: ${self._format_number(total_revenue)}
Gross Profit: ${self._format_number(gross_profit)}
Operating Income: ${self._format_number(operating_income)}
Net Income: ${self._format_number(net_income)}
"""
            return income_summary
        except Exception as e:
            return f"Error retrieving income statement: {str(e)}"

    def _get_balance_sheet(self, stock: yf.Ticker) -> str:
        try:
            # Get the balance sheet (annual)
            balance_sheet = stock.balance_sheet
            
            if balance_sheet.empty:
                return "Balance sheet data not available"
            
            # Most recent year is the last column
            latest_year = balance_sheet.columns[0]
            year = latest_year.year
            
            # Extract key metrics
            total_assets = balance_sheet.loc['Total Assets', latest_year] if 'Total Assets' in balance_sheet.index else None
            total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest', latest_year] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else None
            total_equity = balance_sheet.loc['Total Equity Gross Minority Interest', latest_year] if 'Total Equity Gross Minority Interest' in balance_sheet.index else None
            cash = balance_sheet.loc['Cash And Cash Equivalents', latest_year] if 'Cash And Cash Equivalents' in balance_sheet.index else None
            short_term_debt = balance_sheet.loc['Current Debt', latest_year] if 'Current Debt' in balance_sheet.index else None
            long_term_debt = balance_sheet.loc['Long Term Debt', latest_year] if 'Long Term Debt' in balance_sheet.index else None
            
            # Format the output
            balance_summary = f"""
### Balance Sheet ({year})
Total Assets: ${self._format_number(total_assets)}
Total Liabilities: ${self._format_number(total_liabilities)}
Total Equity: ${self._format_number(total_equity)}
Cash and Equivalents: ${self._format_number(cash)}
Short-term Debt: ${self._format_number(short_term_debt)}
Long-term Debt: ${self._format_number(long_term_debt)}
"""
            return balance_summary
        except Exception as e:
            return f"Error retrieving balance sheet: {str(e)}"

    def _get_cash_flow(self, stock: yf.Ticker) -> str:
        try:
            # Get the cash flow statement (annual)
            cash_flow = stock.cashflow
            
            if cash_flow.empty:
                return "Cash flow data not available"
            
            # Most recent year is the last column
            latest_year = cash_flow.columns[0]
            year = latest_year.year
            
            # Extract key metrics
            operating_cash_flow = cash_flow.loc['Operating Cash Flow', latest_year] if 'Operating Cash Flow' in cash_flow.index else None
            capital_expenditures = cash_flow.loc['Capital Expenditure', latest_year] if 'Capital Expenditure' in cash_flow.index else None
            
            # Calculate free cash flow
            if operating_cash_flow is not None and capital_expenditures is not None:
                free_cash_flow = operating_cash_flow + capital_expenditures  # Capital expenditures are typically negative
            else:
                free_cash_flow = None
                
            dividends_paid = cash_flow.loc['Dividends Paid', latest_year] if 'Dividends Paid' in cash_flow.index else None
            
            # Format the output
            cash_flow_summary = f"""
### Cash Flow ({year})
Operating Cash Flow: ${self._format_number(operating_cash_flow)}
Capital Expenditures: ${self._format_number(capital_expenditures)}
Free Cash Flow: ${self._format_number(free_cash_flow)}
Dividends Paid: ${self._format_number(dividends_paid)}
"""
            return cash_flow_summary
        except Exception as e:
            return f"Error retrieving cash flow statement: {str(e)}"

    def _calculate_ratios(self, stock: yf.Ticker) -> str:
        try:
            # Get necessary financial statements
            income_stmt = stock.income_stmt
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            
            if income_stmt.empty or balance_sheet.empty:
                return "Financial data not available to calculate ratios"
            
            # Most recent year
            latest_year = income_stmt.columns[0]
            year = latest_year.year
            
            # Extract data for ratio calculations
            total_revenue = income_stmt.loc['Total Revenue', latest_year] if 'Total Revenue' in income_stmt.index else None
            gross_profit = income_stmt.loc['Gross Profit', latest_year] if 'Gross Profit' in income_stmt.index else None
            net_income = income_stmt.loc['Net Income', latest_year] if 'Net Income' in income_stmt.index else None
            total_assets = balance_sheet.loc['Total Assets', latest_year] if 'Total Assets' in balance_sheet.index else None
            total_equity = balance_sheet.loc['Total Equity Gross Minority Interest', latest_year] if 'Total Equity Gross Minority Interest' in balance_sheet.index else None
            total_current_assets = balance_sheet.loc['Current Assets', latest_year] if 'Current Assets' in balance_sheet.index else None
            total_current_liabilities = balance_sheet.loc['Current Liabilities', latest_year] if 'Current Liabilities' in balance_sheet.index else None
            short_term_debt = balance_sheet.loc['Current Debt', latest_year] if 'Current Debt' in balance_sheet.index else 0
            long_term_debt = balance_sheet.loc['Long Term Debt', latest_year] if 'Long Term Debt' in balance_sheet.index else 0
            
            # Cash flow ratios
            operating_cash_flow = cash_flow.loc['Operating Cash Flow', latest_year] if not cash_flow.empty and 'Operating Cash Flow' in cash_flow.index else None
            capital_expenditures = cash_flow.loc['Capital Expenditure', latest_year] if not cash_flow.empty and 'Capital Expenditure' in cash_flow.index else None
            
            # Calculate key financial ratios
            try:
                gross_margin = (gross_profit / total_revenue) * 100 if gross_profit is not None and total_revenue is not None else None
            except ZeroDivisionError:
                gross_margin = None
                
            try:
                net_margin = (net_income / total_revenue) * 100 if net_income is not None and total_revenue is not None else None
            except ZeroDivisionError:
                net_margin = None
                
            try:
                roa = (net_income / total_assets) * 100 if net_income is not None and total_assets is not None else None
            except ZeroDivisionError:
                roa = None
                
            try:
                roe = (net_income / total_equity) * 100 if net_income is not None and total_equity is not None else None
            except ZeroDivisionError:
                roe = None
                
            try:
                current_ratio = total_current_assets / total_current_liabilities if total_current_assets is not None and total_current_liabilities is not None else None
            except ZeroDivisionError:
                current_ratio = None
                
            try:
                debt_to_equity = (short_term_debt + long_term_debt) / total_equity if total_equity is not None else None
            except ZeroDivisionError:
                debt_to_equity = None
                
            try:
                if operating_cash_flow is not None and capital_expenditures is not None and total_revenue is not None:
                    fcf = operating_cash_flow + capital_expenditures  # Capital expenditures are typically negative
                    fcf_to_revenue = (fcf / total_revenue) * 100
                else:
                    fcf_to_revenue = None
            except ZeroDivisionError:
                fcf_to_revenue = None
                
            # Format the ratios
            ratios_summary = f"""
### Key Financial Ratios ({year})

#### Profitability
- Gross Margin: {self._format_percentage(gross_margin)}
- Net Profit Margin: {self._format_percentage(net_margin)}
- Return on Assets (ROA): {self._format_percentage(roa)}
- Return on Equity (ROE): {self._format_percentage(roe)}

#### Liquidity
- Current Ratio: {self._format_number(current_ratio) if current_ratio is not None else 'N/A'}

#### Leverage
- Debt-to-Equity Ratio: {self._format_number(debt_to_equity) if debt_to_equity is not None else 'N/A'}

#### Cash Flow
- Free Cash Flow to Revenue: {self._format_percentage(fcf_to_revenue)}
"""
            return ratios_summary
        except Exception as e:
            return f"Error calculating financial ratios: {str(e)}"

    def _get_price_data(self, stock: yf.Ticker, period: str = "1y") -> str:
        try:
            # Get historical market data
            hist = stock.history(period=period)
            
            if hist.empty:
                return "Price data not available"
            
            # Most recent trading day
            latest_date = hist.index[-1]
            
            # Extract key price information
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else None
            
            # Calculate change
            if prev_close is not None:
                price_change = current_price - prev_close
                percent_change = (price_change / prev_close) * 100
                change_str = f"${price_change:.2f} ({percent_change:.2f}%)"
            else:
                change_str = "N/A"
            
            # Calculate highs and lows
            period_high = hist['High'].max()
            period_low = hist['Low'].min()
            
            # Calculate average volume
            avg_volume = hist['Volume'].mean()
            
            # Format the output
            price_summary = f"""
### Recent Price Information (as of {latest_date.strftime('%Y-%m-%d')})
Current Price: ${current_price:.2f}
Daily Change: {change_str}
{period} High: ${period_high:.2f}
{period} Low: ${period_low:.2f}
Average Volume: {self._format_number(avg_volume)}
"""
            return price_summary
        except Exception as e:
            return f"Error retrieving price data: {str(e)}"

    def _format_number(self, value: Any) -> str:
        if value is None or pd.isna(value):
            return 'N/A'
            
        try:
            num = float(value)
            # Format in billions for large numbers
            if abs(num) >= 1e9:
                return f"{num/1e9:.2f}B"
            # Format in millions for medium numbers
            elif abs(num) >= 1e6:
                return f"{num/1e6:.2f}M"
            # Format in thousands for smaller numbers
            elif abs(num) >= 1e3:
                return f"{num/1e3:.2f}K"
            else:
                return f"{num:.2f}"
        except (ValueError, TypeError):
            return str(value)
            
    def _format_percentage(self, value: Optional[float]) -> str:
        if value is None or pd.isna(value):
            return 'N/A'
        return f"{value:.2f}%"