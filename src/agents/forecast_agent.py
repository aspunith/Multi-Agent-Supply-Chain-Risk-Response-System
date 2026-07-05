"""Forecast agent: re-forecast at-risk SKUs.

For each SKU named in the RiskAlert, forecasts the horizon with an 80% band and a confidence
derived from backtest error. Emits one ForecastResult per SKU.
"""
from __future__ import annotations

from ..schema import AgentMessage, AgentName, ForecastResult, MessageType, RiskAlert
from ..tools import forecast_demand

HORIZON = 14


def forecast_agent(world: dict, risk_msg: AgentMessage, recorder=None) -> list[AgentMessage]:
    alert: RiskAlert = risk_msg.payload
    demand_df = world["demand"]
    messages: list[AgentMessage] = []

    for sku_id in alert.affected_skus:
        hist = demand_df.loc[demand_df["sku_id"] == sku_id, "demand"].tolist()
        fc = forecast_demand(hist, horizon=HORIZON, season=7)
        result = ForecastResult(
            sku_id=sku_id,
            horizon_days=HORIZON,
            point_forecast=fc["point_forecast"],
            lower_80=fc["lower_80"],
            upper_80=fc["upper_80"],
            method=fc["method"],
            backtest_mape=fc["backtest_mape"] if fc["backtest_mape"] == fc["backtest_mape"] else 0.5,
            confidence=fc["confidence"],
        )
        msg = AgentMessage(
            correlation_id=world["correlation_id"],
            sender=AgentName.FORECAST,
            recipient=AgentName.PLANNER,
            message_type=MessageType.FORECAST_RESULT,
            payload=result,
        )
        messages.append(msg)
        if recorder:
            recorder.log_step("forecast_agent", "forecast_sku",
                              {"sku_id": sku_id, "method": fc["method"],
                               "mape": result.backtest_mape, "confidence": fc["confidence"]})
            recorder.log_message(msg.to_record())
    return messages
