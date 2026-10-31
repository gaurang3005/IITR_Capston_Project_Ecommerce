# Payment-Failure and Retry Policy

When a payment fails the order is not created, and any amount showing as debited is an authorisation hold rather than a completed charge.

- The cart is held for 60 minutes so the customer can retry, and up to 3 retry attempts are allowed on the same cart before it is released.
- An authorisation hold is voided automatically and the money reappears in 5 to 7 business days with no action needed from the customer.
- If a customer sees a debit but no order, support checks the payment reference and, where the charge did settle, either creates the order manually or initiates a refund.
- Repeated failures on one card usually come from the issuing bank's own risk rules, so we suggest another instrument or cash on delivery.
