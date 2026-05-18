//! Generates an ecommerce platform dataset: ~5k Rails-style error messages
//! from a Shopify-like storefront with varying IDs, hex addresses, and timestamps.
//!
//! This simulates data where deduplication is ineffective — every string is
//! technically unique due to embedded runtime values, but the structural
//! templates are highly repetitive. Normalization (stripping hex, IDs, dates)
//! would collapse these to ~25 templates, making dedup effective again.
//!
//! Usage: cargo run --example gen_ecommerce_errors > testdata/ecommerce_errors.txt

use std::fmt::Write;

fn main() {
    let mut rng = SimpleRng::new(42);
    let mut output = String::new();
    let mut count = 0u32;

    // --- Template family 1: undefined method on ViewComponent (most common) ---
    // ~1500 records across 6 path/component combinations
    let component_variants = [
        (
            "product_inventory_path",
            "Shop::Catalog::Components::Product::DetailView",
            "Shop::Catalog::Product",
            product_attrs as fn(&mut SimpleRng) -> String,
        ),
        (
            "cart_checkout_path",
            "Shop::Storefront::Components::Cart::Summary",
            "Shop::Storefront::Cart",
            cart_attrs as fn(&mut SimpleRng) -> String,
        ),
        (
            "order_tracking_path",
            "Shop::Orders::Components::Order::StatusPanel",
            "Shop::Orders::Order",
            order_attrs as fn(&mut SimpleRng) -> String,
        ),
        (
            "customer_addresses_path",
            "Shop::Accounts::Components::Customer::AddressBook",
            "Shop::Accounts::Customer",
            customer_attrs as fn(&mut SimpleRng) -> String,
        ),
        (
            "collection_products_path",
            "Shop::Catalog::Components::Collection::ProductGrid",
            "Shop::Catalog::Collection",
            collection_attrs as fn(&mut SimpleRng) -> String,
        ),
        (
            "wishlist_items_path",
            "Shop::Storefront::Components::Wishlist::ItemList",
            "Shop::Storefront::Wishlist",
            wishlist_attrs as fn(&mut SimpleRng) -> String,
        ),
    ];

    for (path_method, component_class, model_class, attrs_fn) in &component_variants {
        for _ in 0..250 {
            let hex1 = rng.hex16();
            let hex2 = rng.hex16();
            let attrs = attrs_fn(&mut rng);
            writeln!(
                output,
                "undefined method '{}' for #<{}:0x{} @store_id={}, @record=#<{} {}>>",
                path_method,
                component_class,
                hex1,
                rng.range(1000, 9999),
                model_class,
                attrs
            )
            .unwrap();
            count += 1;

            // Some also have a nested view_context with its own hex address
            if count.is_multiple_of(3) {
                writeln!(
                    output,
                    "undefined method '{}' for #<{}:0x{} @store_id={}, @record=#<{} {}>, @view_context=#<ActionView::Base:0x{}>>",
                    path_method, component_class, hex1, rng.range(1000, 9999),
                    model_class, attrs, hex2
                ).unwrap();
                count += 1;
            }
        }
    }

    // --- Template family 2: NoMethodError with different receiver types ---
    // ~800 records
    let receiver_types = [
        ("Shop::Payments::StripeGateway", "charge", "gateway_id"),
        ("Shop::Shipping::RateCalculator", "estimate", "carrier_id"),
        ("Shop::Inventory::StockManager", "reserve", "warehouse_id"),
        ("Shop::Analytics::EventTracker", "track", "session_id"),
    ];

    for (receiver, method, id_field) in &receiver_types {
        for _ in 0..200 {
            let hex = rng.hex16();
            let id = rng.range(100, 99999);
            let ts = rng.timestamp();
            writeln!(
                output,
                "NoMethodError: undefined method '{}' for #<{}:0x{} @{}={}, @created_at=\"{}\", @active={}> Did you mean? {}!",
                method, receiver, hex, id_field, id, ts, rng.bool_str(), method
            ).unwrap();
            count += 1;
        }
    }

    // --- Template family 3: ActionView template errors with deep object graphs ---
    // ~1000 records across 5 templates
    let template_errors = [
        (
            "shop/catalog/products/show.html.erb",
            "Shop::Catalog::Product",
        ),
        (
            "shop/orders/checkout/summary.html.erb",
            "Shop::Orders::Order",
        ),
        (
            "shop/accounts/dashboard/index.html.erb",
            "Shop::Accounts::Customer",
        ),
        (
            "shop/storefront/collections/show.html.erb",
            "Shop::Catalog::Collection",
        ),
        (
            "shop/admin/inventory/edit.html.erb",
            "Shop::Inventory::StockItem",
        ),
    ];

    for (template, model) in &template_errors {
        for _ in 0..200 {
            let hex = rng.hex16();
            let line = rng.range(10, 300);
            let id = rng.range(1, 50000);
            let ts1 = rng.timestamp();
            let ts2 = rng.timestamp();
            writeln!(
                output,
                "ActionView::Template::Error: undefined local variable or method 'formatted_price' for #<{}:0x{} id: {}, created_at: \"{}\", updated_at: \"{}\"> in {} at line {}",
                model, hex, id, ts1, ts2, template, line
            ).unwrap();
            count += 1;
        }
    }

    // --- Template family 4: ActiveRecord association errors ---
    // ~700 records
    let assoc_errors = [
        (
            "Shop::Catalog::Product",
            "variants",
            "Shop::Catalog::Variant",
        ),
        (
            "Shop::Orders::Order",
            "line_items",
            "Shop::Orders::LineItem",
        ),
        (
            "Shop::Accounts::Customer",
            "addresses",
            "Shop::Accounts::Address",
        ),
        (
            "Shop::Catalog::Collection",
            "products",
            "Shop::Catalog::Product",
        ),
        (
            "Shop::Storefront::Cart",
            "cart_items",
            "Shop::Storefront::CartItem",
        ),
        (
            "Shop::Inventory::StockItem",
            "adjustments",
            "Shop::Inventory::Adjustment",
        ),
        (
            "Shop::Payments::Transaction",
            "refunds",
            "Shop::Payments::Refund",
        ),
    ];

    for (owner, assoc, target) in &assoc_errors {
        for _ in 0..100 {
            let hex = rng.hex16();
            let id = rng.range(1, 99999);
            let ts = rng.timestamp();
            writeln!(
                output,
                "ActiveRecord::HasManyProxy: association '{}' on #<{}:0x{} id: {}, created_at: \"{}\"> references {} which does not exist",
                assoc, owner, hex, id, ts, target
            ).unwrap();
            count += 1;
        }
    }

    print!("{}", output);
    eprintln!(
        "Ecommerce errors dataset: {} total records, ~{} structural templates",
        count,
        component_variants.len() * 2
            + receiver_types.len()
            + template_errors.len()
            + assoc_errors.len()
    );
}

fn product_attrs(rng: &mut SimpleRng) -> String {
    format!(
        "id: {}, sku: \"SKU-{}\", name: \"{}\", price_cents: {}, created_at: \"{}\", updated_at: \"{}\", category_id: {}, in_stock: {}",
        rng.range(1, 50000),
        rng.range(10000, 99999),
        pick(rng, &["Wireless Headphones", "Cotton T-Shirt", "Ceramic Mug", "Leather Wallet", "Running Shoes",
                     "Phone Case", "Yoga Mat", "Coffee Beans", "Desk Lamp", "Backpack"]),
        rng.range(499, 29999),
        rng.timestamp(),
        rng.timestamp(),
        rng.range(1, 200),
        rng.bool_str(),
    )
}

fn cart_attrs(rng: &mut SimpleRng) -> String {
    format!(
        "id: {}, token: \"{}\", item_count: {}, total_cents: {}, created_at: \"{}\", updated_at: \"{}\", customer_id: {}, status: \"{}\"",
        rng.range(1, 99999),
        rng.hex8(),
        rng.range(1, 20),
        rng.range(999, 199999),
        rng.timestamp(),
        rng.timestamp(),
        rng.range(1, 50000),
        pick(rng, &["active", "abandoned", "converted"]),
    )
}

fn order_attrs(rng: &mut SimpleRng) -> String {
    format!(
        "id: {}, number: \"ORD-{}\", status: \"{}\", total_cents: {}, created_at: \"{}\", updated_at: \"{}\", customer_id: {}, shipped_at: {}",
        rng.range(1, 99999),
        rng.range(100000, 999999),
        pick(rng, &["pending", "paid", "shipped", "delivered", "refunded"]),
        rng.range(1999, 499999),
        rng.timestamp(),
        rng.timestamp(),
        rng.range(1, 50000),
        if rng.range(0, 2) == 0 { "nil".to_string() } else { format!("\"{}\"", rng.timestamp()) },
    )
}

fn customer_attrs(rng: &mut SimpleRng) -> String {
    format!(
        "id: {}, email: \"customer{}@example.com\", name: \"{}\", created_at: \"{}\", updated_at: \"{}\", orders_count: {}, verified: {}",
        rng.range(1, 50000),
        rng.range(1, 50000),
        pick(rng, &["Alice Johnson", "Bob Smith", "Carol Davis", "Dan Wilson", "Eve Martinez",
                     "Frank Lee", "Grace Kim", "Hank Brown", "Iris Chen", "Jack Taylor"]),
        rng.timestamp(),
        rng.timestamp(),
        rng.range(0, 150),
        rng.bool_str(),
    )
}

fn collection_attrs(rng: &mut SimpleRng) -> String {
    format!(
        "id: {}, slug: \"{}\", title: \"{}\", product_count: {}, created_at: \"{}\", updated_at: \"{}\", published: {}",
        rng.range(1, 5000),
        pick(rng, &["summer-sale", "new-arrivals", "best-sellers", "clearance", "gift-guide",
                     "electronics", "home-garden", "sports", "books", "fashion"]),
        pick(rng, &["Summer Sale", "New Arrivals", "Best Sellers", "Clearance", "Gift Guide",
                     "Electronics", "Home & Garden", "Sports", "Books", "Fashion"]),
        rng.range(5, 500),
        rng.timestamp(),
        rng.timestamp(),
        rng.bool_str(),
    )
}

fn wishlist_attrs(rng: &mut SimpleRng) -> String {
    format!(
        "id: {}, name: \"{}\", item_count: {}, created_at: \"{}\", updated_at: \"{}\", customer_id: {}, public: {}",
        rng.range(1, 30000),
        pick(rng, &["My Wishlist", "Birthday Ideas", "Holiday Gifts", "Home Renovation", "Camping Gear"]),
        rng.range(0, 50),
        rng.timestamp(),
        rng.timestamp(),
        rng.range(1, 50000),
        rng.bool_str(),
    )
}

fn pick<'a>(rng: &mut SimpleRng, options: &[&'a str]) -> &'a str {
    options[rng.range(0, options.len() as u64 - 1) as usize]
}

/// Minimal deterministic PRNG (xorshift64) so we don't need the rand crate.
struct SimpleRng(u64);

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn range(&mut self, min: u64, max: u64) -> u64 {
        min + (self.next() % (max - min + 1))
    }

    fn hex16(&mut self) -> String {
        format!("{:016x}", self.next())
    }

    fn hex8(&mut self) -> String {
        format!("{:08x}{:08x}", self.next() as u32, self.next() as u32)
    }

    fn bool_str(&mut self) -> &'static str {
        if self.next().is_multiple_of(2) {
            "true"
        } else {
            "false"
        }
    }

    fn timestamp(&mut self) -> String {
        let year = 2024 + self.range(0, 2);
        let month = self.range(1, 12);
        let day = self.range(1, 28);
        let hour = self.range(0, 23);
        let min = self.range(0, 59);
        let sec = self.range(0, 59);
        let nano = self.range(100000000, 999999999);
        format!(
            "{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:09} +0000",
            year, month, day, hour, min, sec, nano
        )
    }
}
