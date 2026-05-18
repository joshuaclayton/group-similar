use criterion::{criterion_group, criterion_main, Criterion};
use group_similar::{Config, Threshold};
use std::hint::black_box;

type Template = Box<dyn Fn(&mut SimpleRng) -> String>;

/// Blog CMS dataset: ~20k Java FQCNs with ~35 unique strings.
/// Dedup-heavy — mirrors patterns like log/trace data from a WordPress-like platform.
fn blog_cms_data() -> Vec<String> {
    let methods = [
        ("org.blogengine.api.domain.PostMapper.findById", 1200),
        ("org.blogengine.api.domain.PostMapper.findAll", 1200),
        ("org.blogengine.api.domain.PostMapper.findBySlug", 1200),
        ("org.blogengine.api.domain.PostMapper.create", 1200),
        ("org.blogengine.api.domain.PostMapper.update", 1200),
        ("org.blogengine.api.domain.PostMapper.delete", 1200),
        ("org.blogengine.api.domain.PostMapper.publish", 1200),
        ("org.blogengine.api.domain.PostMapper.findByCategory", 1200),
        ("org.blogengine.api.domain.CommentMapper.findByPostId", 800),
        ("org.blogengine.api.domain.CommentMapper.create", 800),
        ("org.blogengine.api.domain.CommentMapper.delete", 800),
        ("org.blogengine.api.domain.CommentMapper.moderate", 800),
        ("org.blogengine.api.domain.UserMapper.findById", 600),
        ("org.blogengine.api.domain.UserMapper.findByEmail", 600),
        ("org.blogengine.api.domain.UserMapper.authenticate", 600),
        ("org.blogengine.api.domain.UserMapper.updateProfile", 600),
        ("org.blogengine.api.domain.CategoryMapper.findAll", 200),
        ("org.blogengine.api.domain.CategoryMapper.findBySlug", 200),
        ("org.blogengine.api.domain.CategoryMapper.create", 200),
        ("org.blogengine.api.domain.TagMapper.findByPostId", 200),
        ("org.blogengine.api.domain.TagMapper.findOrCreate", 200),
        ("org.blogengine.api.domain.MediaMapper.upload", 200),
        ("org.blogengine.api.domain.MediaMapper.findById", 200),
        ("org.blogengine.api.controllers.PostController.index", 400),
        ("org.blogengine.api.controllers.PostController.show", 400),
        ("org.blogengine.api.controllers.PostController.create", 400),
        (
            "org.blogengine.api.controllers.CommentController.create",
            400,
        ),
        ("org.blogengine.api.controllers.UserController.profile", 400),
        ("org.blogengine.api.controllers.MediaController.upload", 400),
        ("org.blogengine.api.services.PostService.publish", 300),
        ("org.blogengine.api.services.PostService.scheduleDraft", 300),
        ("org.blogengine.api.services.SearchService.indexPost", 300),
        ("org.blogengine.api.services.SearchService.query", 300),
        (
            "org.blogengine.api.services.NotificationService.sendEmail",
            300,
        ),
        ("org.blogengine.api.services.FeedService.generateRss", 300),
    ];

    let mut data = Vec::new();
    for (method, count) in &methods {
        for _ in 0..*count {
            data.push(method.to_string());
        }
    }
    data
}

/// Ecommerce errors dataset: ~25k Rails-style error messages with varying IDs/hex/timestamps.
/// Every string is unique, but structural templates repeat.
/// At production scale (25k records), O(n^2) ≈ 312M pairs. The bench runs a 2k slice
/// to stay tractable on the current algorithm.
fn ecommerce_errors_data() -> Vec<String> {
    let mut rng = SimpleRng::new(42);
    let mut data = Vec::new();

    let templates: Vec<Template> = vec![
        Box::new(|r| {
            format!(
            "undefined method 'product_inventory_path' for #<Shop::Catalog::Components::Product::DetailView:0x{} @store_id={}, @record=#<Shop::Catalog::Product id: {}, sku: \"SKU-{}\", price_cents: {}, created_at: \"{}\", updated_at: \"{}\">>",
            r.hex16(), r.range(1000, 9999), r.range(1, 50000), r.range(10000, 99999), r.range(499, 29999), r.timestamp(), r.timestamp()
        )
        }),
        Box::new(|r| {
            format!(
            "undefined method 'cart_checkout_path' for #<Shop::Storefront::Components::Cart::Summary:0x{} @store_id={}, @record=#<Shop::Storefront::Cart id: {}, item_count: {}, total_cents: {}, created_at: \"{}\", updated_at: \"{}\">>",
            r.hex16(), r.range(1000, 9999), r.range(1, 99999), r.range(1, 20), r.range(999, 199999), r.timestamp(), r.timestamp()
        )
        }),
        Box::new(|r| {
            format!(
            "undefined method 'order_tracking_path' for #<Shop::Orders::Components::Order::StatusPanel:0x{} @store_id={}, @record=#<Shop::Orders::Order id: {}, number: \"ORD-{}\", status: \"paid\", total_cents: {}, created_at: \"{}\">>",
            r.hex16(), r.range(1000, 9999), r.range(1, 99999), r.range(100000, 999999), r.range(1999, 499999), r.timestamp()
        )
        }),
        Box::new(|r| {
            format!(
            "NoMethodError: undefined method 'charge' for #<Shop::Payments::StripeGateway:0x{} @gateway_id={}, @created_at=\"{}\", @active=true>",
            r.hex16(), r.range(100, 99999), r.timestamp()
        )
        }),
        Box::new(|r| {
            format!(
            "ActionView::Template::Error: undefined local variable or method 'formatted_price' for #<Shop::Catalog::Product:0x{} id: {}, created_at: \"{}\", updated_at: \"{}\"> in shop/catalog/products/show.html.erb at line {}",
            r.hex16(), r.range(1, 50000), r.timestamp(), r.timestamp(), r.range(10, 300)
        )
        }),
    ];

    for template in &templates {
        for _ in 0..5000 {
            data.push(template(&mut rng));
        }
    }
    data
}

/// Mixed dataset: ~30k records combining both patterns — heavily duplicated short strings
/// (15-50 chars) and long unique error messages (150-350 chars). The bench runs a 5k slice.
fn mixed_data() -> Vec<String> {
    let mut rng = SimpleRng::new(99);
    let mut data = Vec::new();

    let short_methods = [
        "Shop::ProductService.find",
        "Shop::ProductService.search",
        "Shop::ProductService.create",
        "Shop::ProductService.update",
        "Shop::CartService.add_item",
        "Shop::CartService.remove_item",
        "Shop::CartService.calculate_total",
        "Shop::CartService.apply_discount",
        "Shop::OrderService.place",
        "Shop::OrderService.cancel",
        "Shop::OrderService.refund",
        "Shop::OrderService.ship",
        "Shop::OrderService.deliver",
        "Shop::PaymentService.charge",
        "Shop::PaymentService.authorize",
        "Shop::PaymentService.capture",
        "Shop::PaymentService.void",
        "Shop::CustomerService.register",
        "Shop::CustomerService.authenticate",
        "Shop::CustomerService.update_profile",
        "Shop::CustomerService.reset_password",
        "Shop::InventoryService.check_stock",
        "Shop::InventoryService.reserve",
        "Shop::InventoryService.release",
        "Shop::InventoryService.adjust",
        "Shop::ShippingService.estimate",
        "Shop::ShippingService.create_label",
        "Shop::ShippingService.track",
        "Shop::NotificationService.send_email",
        "Shop::NotificationService.send_sms",
        "Shop::SearchService.index",
        "Shop::SearchService.query",
        "Shop::SearchService.suggest",
        "Shop::AnalyticsService.track_event",
        "Shop::AnalyticsService.page_view",
        "Shop::ReviewService.create",
        "Shop::ReviewService.moderate",
        "Shop::ReviewService.average_rating",
        "Shop::WishlistService.add",
        "Shop::WishlistService.remove",
    ];
    for method in &short_methods {
        let count = rng.range(200, 500) as usize;
        for _ in 0..count {
            data.push(method.to_string());
        }
    }

    let error_templates: Vec<Template> = vec![
        Box::new(|r| {
            format!(
            "undefined method 'product_path' for #<Shop::Catalog::ProductCard:0x{} @id={}, @name=\"{}\", @price_cents={}, @created_at=\"{}\">",
            r.hex16(), r.range(1, 99999), pick(r, &["Widget", "Gadget", "Doohickey", "Thingamajig", "Gizmo"]), r.range(100, 99999), r.timestamp()
        )
        }),
        Box::new(|r| {
            format!(
            "undefined method 'checkout_path' for #<Shop::Orders::CheckoutForm:0x{} @order_id={}, @total_cents={}, @item_count={}, @created_at=\"{}\">",
            r.hex16(), r.range(1, 99999), r.range(1000, 999999), r.range(1, 50), r.timestamp()
        )
        }),
        Box::new(|r| {
            format!(
            "NoMethodError: undefined method 'balance' for #<Shop::Payments::Wallet:0x{} @customer_id={}, @currency=\"USD\", @last_charged_at=\"{}\">",
            r.hex16(), r.range(1, 50000), r.timestamp()
        )
        }),
        Box::new(|r| {
            format!(
            "ActionView::Template::Error: undefined local variable 'shipping_rate' for #<Shop::Shipping::RateCard:0x{} id: {}, carrier: \"{}\", created_at: \"{}\"> in shop/shipping/rates/show.html.erb at line {}",
            r.hex16(), r.range(1, 9999), pick(r, &["USPS", "FedEx", "UPS", "DHL"]), r.timestamp(), r.range(1, 200)
        )
        }),
        Box::new(|r| {
            format!(
            "ActiveRecord::RecordNotFound: Couldn't find Shop::Catalog::Product with id={} [WHERE shop_id={} AND deleted_at IS NULL] at \"{}\"",
            r.range(1, 99999), r.range(1, 9999), r.timestamp()
        )
        }),
        Box::new(|r| {
            format!(
            "ArgumentError: wrong number of arguments (given {}, expected {}) in Shop::Inventory::StockManager#reserve at 0x{}",
            r.range(0, 5), r.range(1, 4), r.hex16()
        )
        }),
        Box::new(|r| {
            format!(
            "Redis::TimeoutError: Connection timed out after {}ms for Shop::Cache::ProductLookup key=\"product:{}\" at \"{}\"",
            r.range(100, 5000), r.range(1, 99999), r.timestamp()
        )
        }),
        Box::new(|r| {
            format!(
            "Shop::RateLimitExceeded: customer_id={} exceeded {} requests/min on endpoint=/api/v2/products at \"{}\"",
            r.range(1, 50000), r.range(60, 1000), r.timestamp()
        )
        }),
        Box::new(|r| {
            format!(
            "Stripe::CardError: card_id=card_{} declined with code={} for charge of {} cents on order ORD-{} at \"{}\"",
            r.hex16(), pick(r, &["insufficient_funds", "expired_card", "processing_error", "incorrect_cvc"]),
            r.range(100, 999999), r.range(100000, 999999), r.timestamp()
        )
        }),
        Box::new(|r| {
            format!(
            "Shop::Fulfillment::ShipmentError: shipment_id={} tracking={} carrier=\"{}\" status=\"failed\" reason=\"{}\" at \"{}\"",
            r.range(1, 99999), r.hex16(),
            pick(r, &["USPS", "FedEx", "UPS", "DHL"]),
            pick(r, &["address_invalid", "package_too_large", "customs_hold", "carrier_delay"]),
            r.timestamp()
        )
        }),
    ];
    for template in &error_templates {
        for _ in 0..1500 {
            data.push(template(&mut rng));
        }
    }

    data
}

fn pick<'a>(rng: &mut SimpleRng, options: &[&'a str]) -> &'a str {
    options[rng.range(0, options.len() as u64 - 1) as usize]
}

fn bench_blog_cms(c: &mut Criterion) {
    let data = blog_cms_data();
    let strings: Vec<&str> = data.iter().map(|s| s.as_str()).collect();

    let mut group = c.benchmark_group("blog_cms");
    group.sample_size(10);
    group.bench_function("group_similar", |b| {
        let config: Config<&str> = Config::jaro_winkler(Threshold::default());
        b.iter(|| group_similar::group_similar(black_box(&strings), &config))
    });
    group.finish();
}

fn bench_ecommerce_errors(c: &mut Criterion) {
    let data = ecommerce_errors_data();
    let strings: Vec<&str> = data.iter().map(|s| s.as_str()).collect();

    // O(25k²) ≈ 312M pairs at full size. Use a 2k slice to keep bench runtime
    // tractable on the current algorithm; matches the perf-2 slice for apples-to-apples.
    let small_slice = &strings[..2000];
    let mut group = c.benchmark_group("ecommerce_25k");
    group.sample_size(10);
    group.bench_function("no_normalizer_2k_slice", |b| {
        let config: Config<&str> = Config::jaro_winkler(Threshold::default());
        b.iter(|| group_similar::group_similar(black_box(small_slice), &config))
    });
    group.finish();
}

fn bench_mixed(c: &mut Criterion) {
    let data = mixed_data();
    let strings: Vec<&str> = data.iter().map(|s| s.as_str()).collect();
    let unique_count = {
        let mut u: Vec<&str> = strings.clone();
        u.sort_unstable();
        u.dedup();
        u.len()
    };
    eprintln!(
        "mixed dataset: {} total, {} unique",
        strings.len(),
        unique_count
    );

    // Without normalization, the ~15k long error messages remain unique → large matrix.
    // Use a 5k slice to keep tractable; matches the perf-2 slice for apples-to-apples.
    let small_slice = &strings[..5000.min(strings.len())];
    let mut group = c.benchmark_group("mixed_30k");
    group.sample_size(10);
    group.bench_function("no_normalizer_5k_slice", |b| {
        let config: Config<&str> = Config::jaro_winkler(Threshold::default());
        b.iter(|| group_similar::group_similar(black_box(small_slice), &config))
    });
    group.finish();
}

fn bench_similarity_matrix(c: &mut Criterion) {
    let list = black_box(&[
        "A",
        "a",
        "aa",
        "aal",
        "aalii",
        "aam",
        "Aani",
        "aardvark",
        "aardwolf",
        "Aaron",
        "Aaronic",
        "Aaronical",
        "Aaronite",
        "Aaronitic",
        "Aaru",
        "Ab",
        "aba",
        "Ababdeh",
        "Ababua",
        "abac",
        "abaca",
        "abacate",
        "abacay",
        "abacinate",
        "abacination",
        "abaciscus",
        "abacist",
        "aback",
        "abactinal",
        "abactinally",
        "abaction",
        "abactor",
        "abaculus",
        "abacus",
        "Abadite",
        "abaff",
        "abaft",
        "abaisance",
        "abaiser",
        "abaissed",
        "abalienate",
        "abalienation",
        "abalone",
        "Abama",
        "abampere",
        "abandon",
        "abandonable",
        "abandoned",
        "abandonedly",
        "abandonee",
        "abandoner",
        "abandonment",
        "Abanic",
        "Abantes",
        "abaptiston",
        "Abarambo",
        "Abaris",
        "abarthrosis",
        "abarticular",
        "abarticulation",
        "abas",
        "abase",
        "abased",
        "abasedly",
        "abasedness",
        "abasement",
        "abaser",
        "Abasgi",
        "abash",
        "abashed",
        "abashedly",
        "abashedness",
        "abashless",
        "abashlessly",
        "abashment",
        "abasia",
        "abasic",
        "abask",
        "Abassin",
        "abastardize",
        "abatable",
        "abate",
        "abatement",
        "abater",
        "abatis",
        "abatised",
        "abaton",
        "abator",
        "abattoir",
        "Abatua",
        "abature",
        "abave",
        "abaxial",
        "abaxile",
        "abaze",
        "abb",
        "Abba",
        "abbacomes",
        "abbacy",
        "Abbadide",
    ]);

    c.bench_function("similarity_matrix/dictionary_100", |b| {
        let compare = |a: &&str, b: &&str| 1.0 - jaro_winkler::jaro_winkler(a, b);
        b.iter(|| group_similar::similarity_matrix(list, &compare))
    });
}

criterion_group!(
    benches,
    bench_similarity_matrix,
    bench_blog_cms,
    bench_ecommerce_errors,
    bench_mixed
);
criterion_main!(benches);

// --- Minimal PRNG for deterministic benchmark data ---

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

    fn timestamp(&mut self) -> String {
        format!(
            "{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:09} +0000",
            2024 + self.range(0, 2),
            self.range(1, 12),
            self.range(1, 28),
            self.range(0, 23),
            self.range(0, 59),
            self.range(0, 59),
            self.range(100000000, 999999999)
        )
    }
}
