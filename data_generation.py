# Standard library imports
import random
import datetime

# Third-party imports
import pandas as pd
from faker import Faker

# Initialize Faker
fake = Faker()

# Define sample data
products = ["Laptop", "Smartphone", "Tablet", "Headphones", "Monitor", "Keyboard", "Mouse"]
regions = ["North America", "Europe", "Asia", "South America", "Australia", "Africa"]
payment_methods = ["Credit Card", "Debit Card", "PayPal", "Cash", "Bank Transfer"]


# Generate random sales data
def generate_sales_data(num_records=200):
    data = []
    for _ in range(num_records):
        timestamp = datetime.datetime.now() - datetime.timedelta(days=random.randint(0, 30))
        product = random.choice(products)
        quantity = random.randint(1, 10)
        price = round(random.uniform(50, 1500), 2)
        total_amount = round(price * quantity, 2)
        discount = round(total_amount * random.uniform(0, 0.2), 2)  # Up to 20% discount
        final_amount = round(total_amount - discount, 2)
        customer_name = fake.name()  # Generate realistic customer name
        customer_id = fake.uuid4()  # Generate unique Customer ID
        region = random.choice(regions)
        payment_method = random.choice(payment_methods)

        data.append([
            timestamp, product, quantity, price, total_amount, discount, final_amount,
            customer_id, customer_name, region, payment_method
        ])

    df = pd.DataFrame(data, columns=[
        "Timestamp", "Product", "Quantity", "Price", "Total Price", "Discount Applied",
        "Final Price", "Customer ID", "Customer Name", "Region", "Payment Method"
    ])
    return df


# Generate and save to CSV
sales_data = generate_sales_data(100)
sales_data.to_csv("sales_data.csv", index=False)

print(
    "Expanded sales data generated with Faker names and unique Customer IDs, saved to 'expanded_sales_data_faker.csv'."
)
