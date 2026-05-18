//! Generates a blog CMS dataset: ~25k Java-style fully-qualified method names
//! from a WordPress-like platform with only ~35 unique strings.
//!
//! This simulates data where deduplication is extremely effective — many callers
//! invoke the same methods, producing massive repetition in logs/traces.
//!
//! Usage: cargo run --example gen_blog_cms > testdata/blog_cms.txt

fn main() {
    let methods = [
        // PostMapper — the workhorse, most frequently called
        "org.blogengine.api.domain.PostMapper.findById",
        "org.blogengine.api.domain.PostMapper.findAll",
        "org.blogengine.api.domain.PostMapper.findBySlug",
        "org.blogengine.api.domain.PostMapper.create",
        "org.blogengine.api.domain.PostMapper.update",
        "org.blogengine.api.domain.PostMapper.delete",
        "org.blogengine.api.domain.PostMapper.publish",
        "org.blogengine.api.domain.PostMapper.findByCategory",
        // CommentMapper
        "org.blogengine.api.domain.CommentMapper.findByPostId",
        "org.blogengine.api.domain.CommentMapper.create",
        "org.blogengine.api.domain.CommentMapper.delete",
        "org.blogengine.api.domain.CommentMapper.moderate",
        // UserMapper
        "org.blogengine.api.domain.UserMapper.findById",
        "org.blogengine.api.domain.UserMapper.findByEmail",
        "org.blogengine.api.domain.UserMapper.authenticate",
        "org.blogengine.api.domain.UserMapper.updateProfile",
        // CategoryMapper
        "org.blogengine.api.domain.CategoryMapper.findAll",
        "org.blogengine.api.domain.CategoryMapper.findBySlug",
        "org.blogengine.api.domain.CategoryMapper.create",
        // TagMapper
        "org.blogengine.api.domain.TagMapper.findByPostId",
        "org.blogengine.api.domain.TagMapper.findOrCreate",
        // MediaMapper
        "org.blogengine.api.domain.MediaMapper.upload",
        "org.blogengine.api.domain.MediaMapper.findById",
        // Controllers
        "org.blogengine.api.controllers.PostController.index",
        "org.blogengine.api.controllers.PostController.show",
        "org.blogengine.api.controllers.PostController.create",
        "org.blogengine.api.controllers.CommentController.create",
        "org.blogengine.api.controllers.UserController.profile",
        "org.blogengine.api.controllers.MediaController.upload",
        // Services
        "org.blogengine.api.services.PostService.publish",
        "org.blogengine.api.services.PostService.scheduleDraft",
        "org.blogengine.api.services.SearchService.indexPost",
        "org.blogengine.api.services.SearchService.query",
        "org.blogengine.api.services.NotificationService.sendEmail",
        "org.blogengine.api.services.FeedService.generateRss",
    ];

    // Weights: PostMapper methods are called far more often than others,
    // reflecting real-world hot paths in a blog platform.
    let weights: Vec<(&str, usize)> = methods
        .iter()
        .map(|m| {
            let w = if m.contains("PostMapper") {
                1200
            } else if m.contains("CommentMapper") {
                800
            } else if m.contains("UserMapper") {
                600
            } else if m.contains("Controller") {
                400
            } else if m.contains("Service") {
                300
            } else {
                200
            };
            (*m, w)
        })
        .collect();

    let total: usize = weights.iter().map(|(_, w)| w).sum();
    eprintln!(
        "Blog CMS dataset: {} unique methods, {} total records",
        methods.len(),
        total
    );

    for (method, count) in &weights {
        for _ in 0..*count {
            println!("{}", method);
        }
    }
}
