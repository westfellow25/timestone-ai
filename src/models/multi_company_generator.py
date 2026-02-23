"""
Multi-Company Digital Twin Generator

Creates digital twins for companies across different industries
"""

import sys
sys.path.append('/workspaces/timestone-ai')

from src.models.digital_twin import DigitalTwin, CompanyMetrics
from src.models.industry_templates import Industry, INDUSTRY_TEMPLATES
import random
import json


class MultiCompanyGenerator:
    """Generate digital twins for multiple companies"""
    
    def __init__(self):
        self.companies = []
    
    def generate_company(self, name: str, industry: Industry, scale: str = "medium") -> DigitalTwin:
        """
        Generate a realistic digital twin for a company
        
        Args:
            name: Company name
            industry: Industry type
            scale: "small", "medium", "large"
        """
        template = INDUSTRY_TEMPLATES[industry]
        
        # Scale factors
        scale_factors = {
            "small": 0.1,
            "medium": 0.5,
            "large": 0.9
        }
        factor = scale_factors.get(scale, 0.5)
        
        # Generate realistic metrics
        revenue_range = template.typical_revenue_range
        revenue = revenue_range[0] + (revenue_range[1] - revenue_range[0]) * factor
        
        margin_range = template.typical_margin_range
        margin = margin_range[0] + (margin_range[1] - margin_range[0]) * random.uniform(0.3, 0.7)
        
        operating_costs = revenue * (1 - margin)
        
        # Estimate employee count based on revenue
        revenue_per_employee = random.uniform(200_000, 500_000)
        employee_count = int(revenue / revenue_per_employee)
        
        # Market share varies by scale
        market_share_base = {
            "small": (0.01, 0.05),
            "medium": (0.05, 0.20),
            "large": (0.20, 0.60)
        }
        market_share = random.uniform(*market_share_base[scale])
        
        # Customer metrics
        avg_transaction = revenue / random.uniform(10000, 100000)
        customer_count = int(revenue / avg_transaction)
        
        # Growth rate
        growth_rate = random.uniform(0.03, 0.15)
        
        metrics = CompanyMetrics(
            revenue=revenue,
            operating_costs=operating_costs,
            employee_count=employee_count,
            market_share=market_share,
            customer_count=customer_count,
            avg_transaction_value=avg_transaction,
            growth_rate=growth_rate,
            industry=industry.value
        )
        
        twin = DigitalTwin(
            company_name=name,
            metrics=metrics,
            metadata={
                "industry": industry.value,
                "scale": scale,
                "template_used": True
            }
        )
        
        self.companies.append(twin)
        return twin
    
    def save_all(self, directory: str = "."):
        """Save all company twins"""
        for company in self.companies:
            filename = f"{directory}/{company.company_name.lower().replace(' ', '_')}_twin.json"
            company.save(filename)
            print(f"Saved: {filename}")


# Predefined companies
COMPANIES_LIBRARY = [
    # Kazakhstan Companies
    ("Kazakhstan Temir Zholy (KTZ)", Industry.TRANSPORTATION, "large"),
    ("KEGOC (Kazakhstan Electricity Grid)", Industry.ENERGY, "large"),
    ("Kaspi.kz", Industry.FINTECH, "large"),
    
    # International Templates
    ("TechFlow SaaS", Industry.SAAS, "medium"),
    ("PrecisionMfg Inc", Industry.MANUFACTURING, "medium"),
]


if __name__ == "__main__":
    print("Generating Multi-Industry Digital Twins")
    print("=" * 70)
    
    generator = MultiCompanyGenerator()
    
    for name, industry, scale in COMPANIES_LIBRARY:
        print(f"\nGenerating: {name}")
        twin = generator.generate_company(name, industry, scale)
        print(f"  Industry: {industry.value}")
        print(f"  Revenue: ${twin.metrics.revenue:,.0f}")
        print(f"  Employees: {twin.metrics.employee_count:,}")
        print(f"  Profit Margin: {twin.metrics.profit_margin:.1%}")
    
    print(f"\n{'=' * 70}")
    print(f"Generated {len(generator.companies)} company digital twins")
    
    # Save all
    generator.save_all()
    print(f"\n✅ All twins saved!")
