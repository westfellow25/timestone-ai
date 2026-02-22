"""
Digital Twin Model for Business Transformation Simulation

This module creates a synthetic representation of a company
based on financial, operational, and market data.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class CompanyMetrics:
    """Core metrics representing company state"""
    revenue: float
    operating_costs: float
    employee_count: int
    market_share: float
    customer_count: int
    avg_transaction_value: float
    growth_rate: float
    industry: str
    
    @property
    def profit_margin(self) -> float:
        """Calculate profit margin"""
        if self.revenue == 0:
            return 0.0
        return (self.revenue - self.operating_costs) / self.revenue
    
    @property
    def revenue_per_employee(self) -> float:
        """Calculate revenue per employee"""
        if self.employee_count == 0:
            return 0.0
        return self.revenue / self.employee_count


class DigitalTwin:
    """
    Digital Twin of a company for transformation simulation
    
    Creates a synthetic model that can be used to test
    transformation scenarios without real-world risk.
    """
    
    def __init__(
        self,
        company_name: str,
        metrics: CompanyMetrics,
        metadata: Optional[Dict] = None
    ):
        self.company_name = company_name
        self.metrics = metrics
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.version = "1.0.0"
    
    def simulate_transformation(
        self,
        transformation_type: str,
        parameters: Dict
    ) -> Dict:
        """
        Simulate a transformation scenario
        
        Args:
            transformation_type: Type of transformation (e.g., "digital", "pricing")
            parameters: Transformation-specific parameters
            
        Returns:
            Simulated outcome metrics
        """
        # This will be expanded with actual simulation logic
        return {
            "success": True,
            "message": f"Simulation of {transformation_type} transformation",
            "baseline_metrics": self.get_baseline(),
            "projected_metrics": self._project_metrics(transformation_type, parameters)
        }
    
    def get_baseline(self) -> Dict:
        """Get current baseline metrics"""
        return {
            "revenue": self.metrics.revenue,
            "profit_margin": self.metrics.profit_margin,
            "market_share": self.metrics.market_share,
            "employee_productivity": self.metrics.revenue_per_employee
        }
    
    def _project_metrics(
        self,
        transformation_type: str,
        parameters: Dict
    ) -> Dict:
        """
        Project metrics after transformation
        
        This is a simplified placeholder. Real implementation
        will use Monte Carlo simulation and historical data.
        """
        baseline = self.get_baseline()
        
        # Simple projection logic (to be replaced with ML models)
        impact_factor = parameters.get("impact_factor", 1.1)
        
        return {
            "revenue": baseline["revenue"] * impact_factor,
            "profit_margin": baseline["profit_margin"] * 1.05,
            "market_share": min(baseline["market_share"] * 1.02, 1.0),
            "employee_productivity": baseline["employee_productivity"] * impact_factor
        }
    
    def to_dict(self) -> Dict:
        """Export twin as dictionary"""
        return {
            "company_name": self.company_name,
            "metrics": {
                "revenue": self.metrics.revenue,
                "operating_costs": self.metrics.operating_costs,
                "employee_count": self.metrics.employee_count,
                "market_share": self.metrics.market_share,
                "customer_count": self.metrics.customer_count,
                "avg_transaction_value": self.metrics.avg_transaction_value,
                "growth_rate": self.metrics.growth_rate,
                "industry": self.metrics.industry
            },
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "version": self.version
        }
    
    def save(self, filepath: str):
        """Save digital twin to file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'DigitalTwin':
        """Load digital twin from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        metrics = CompanyMetrics(**data['metrics'])
        twin = cls(
            company_name=data['company_name'],
            metrics=metrics,
            metadata=data.get('metadata', {})
        )
        twin.version = data.get('version', '1.0.0')
        return twin


# Example usage
if __name__ == "__main__":
    # Create a digital twin for KTZ (Kazakhstan Railways)
    ktz_metrics = CompanyMetrics(
        revenue=500_000_000,  # $500M annual revenue
        operating_costs=450_000_000,
        employee_count=10_000,
        market_share=0.85,  # 85% of national rail market
        customer_count=50_000,
        avg_transaction_value=10_000,
        growth_rate=0.05,  # 5% annual growth
        industry="Transportation & Logistics"
    )
    
    ktz_twin = DigitalTwin(
        company_name="Kazakhstan Temir Zholy (KTZ)",
        metrics=ktz_metrics,
        metadata={
            "country": "Kazakhstan",
            "founded": 2002,
            "type": "National Railway Operator"
        }
    )
    
    print(f"Created digital twin: {ktz_twin.company_name}")
    print(f"Baseline profit margin: {ktz_twin.metrics.profit_margin:.2%}")
    print(f"Revenue per employee: ${ktz_twin.metrics.revenue_per_employee:,.0f}")
    
    # Simulate dynamic pricing transformation
    result = ktz_twin.simulate_transformation(
        transformation_type="dynamic_pricing",
        parameters={"impact_factor": 1.22}  # 22% revenue increase
    )
    
    print("\nSimulation Result:")
    print(f"Baseline Revenue: ${result['baseline_metrics']['revenue']:,.0f}")
    print(f"Projected Revenue: ${result['projected_metrics']['revenue']:,.0f}")
    
    # Save twin
    ktz_twin.save("ktz_digital_twin.json")
    print("\nDigital twin saved to ktz_digital_twin.json")
