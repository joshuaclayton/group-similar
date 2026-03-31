//! Built-in string normalizers used in the dedup stage.
//!
//! A normalizer is any `Fn(&str) -> String`. It produces the canonical key
//! used to group equivalent records during dedup; original values are
//! preserved in the final output.
//!
//! # Recommended entry point
//!
//! [`default_normalizer`] fuses the four built-in patterns into a single-pass
//! scanner: `0x`-prefixed hex addresses, bare hex tokens, quoted timestamps,
//! and decimal integer sequences are each replaced with a fixed placeholder.
//! Inputs with no digits are returned untouched.
//!
//! # Custom pipelines
//!
//! For finer control, the individual pattern functions ([`hex_addresses`],
//! [`numeric_ids`], [`timestamps`], [`hex_tokens`]) can be combined with
//! [`compose`] or [`compose_all`]. Use [`identity`] when no normalization is
//! desired.
//!
//! # Internal layout
//!
//! The implementation is split into two private submodules: [`engine`]
//! provides the byte-level multi-matcher scanner shared by all patterns, and
//! [`patterns`] holds one submodule per pattern, each exposing an
//! `is_start` / `consume` pair that the engine dispatches through.

use engine::{replace_pattern, replace_patterns, Matcher};

/// Identity normalizer — returns the input unchanged.
pub fn identity(input: &str) -> String {
    input.to_string()
}

/// The recommended default normalizer for structured data with embedded
/// runtime values (hex addresses, IDs, timestamps, tokens).
///
/// Uses a fused single-pass scanner (one allocation, one scan) rather than
/// chaining individual normalizers. Pattern priority order:
/// 1. `hex_addresses` — match `0x`-prefixed hex first so later patterns don't split them
/// 2. `hex_tokens` — match bare hex tokens (8+ hex chars) before `numeric_ids` would consume their digit runs
/// 3. `timestamps` — match quoted `"YYYY-MM-DD ..."` timestamps before `numeric_ids` would replace the digits inside
/// 4. `numeric_ids` — most permissive, checked last to catch remaining standalone digit sequences
pub fn default_normalizer() -> Box<dyn Fn(&str) -> String + Send + Sync> {
    const MATCHERS: &[Matcher] = &[
        (
            patterns::hex_addresses::is_start,
            patterns::hex_addresses::consume,
        ),
        (
            patterns::hex_tokens::is_start,
            patterns::hex_tokens::consume,
        ),
        (
            patterns::timestamps::is_start,
            patterns::timestamps::consume,
        ),
        (
            patterns::numeric_ids::is_start,
            patterns::numeric_ids::consume,
        ),
    ];

    Box::new(|input| {
        // All patterns require ASCII digits; strings without digits skip scanning.
        if !input.bytes().any(|b| b.is_ascii_digit()) {
            return input.to_string();
        }
        replace_patterns(input, MATCHERS)
    })
}

/// Compose two normalizers into one that applies `first`, then `second`.
pub fn compose<F, G>(first: F, second: G) -> Box<dyn Fn(&str) -> String + Send + Sync>
where
    F: Fn(&str) -> String + Send + Sync + 'static,
    G: Fn(&str) -> String + Send + Sync + 'static,
{
    Box::new(move |input| second(&first(input)))
}

/// Compose a list of normalizers left-to-right.
pub fn compose_all(
    normalizers: Vec<Box<dyn Fn(&str) -> String + Send + Sync>>,
) -> Box<dyn Fn(&str) -> String + Send + Sync> {
    Box::new(move |input| {
        let mut result = input.to_string();
        for f in &normalizers {
            result = f(&result);
        }
        result
    })
}

/// Replace hex addresses (e.g. `0x00007f3a1b2c3d48`) with a fixed placeholder.
pub fn hex_addresses(input: &str) -> String {
    replace_pattern(
        input,
        patterns::hex_addresses::is_start,
        patterns::hex_addresses::consume,
    )
}

/// Replace decimal integer sequences (e.g. `12345`) with `0`.
pub fn numeric_ids(input: &str) -> String {
    replace_pattern(
        input,
        patterns::numeric_ids::is_start,
        patterns::numeric_ids::consume,
    )
}

/// Replace quoted timestamps matching `"YYYY-MM-DD HH:MM:SS..."` with a placeholder.
pub fn timestamps(input: &str) -> String {
    replace_pattern(
        input,
        patterns::timestamps::is_start,
        patterns::timestamps::consume,
    )
}

/// Replace hex sequences of 8+ hex chars not preceded by `0x` (e.g. object tokens, hashes).
pub fn hex_tokens(input: &str) -> String {
    replace_pattern(
        input,
        patterns::hex_tokens::is_start,
        patterns::hex_tokens::consume,
    )
}

/// Byte-level scanner that dispatches through a list of `(is_start, consume)`
/// matchers. All patterns share this engine so they can be fused into a
/// single-pass scan.
mod engine {
    pub(super) type Matcher = (
        fn(&[u8], usize) -> bool,
        fn(&[u8], usize) -> Option<(usize, &'static str)>,
    );

    /// Single-pass scanner that tries multiple pattern matchers at each byte
    /// position. First matching pattern wins. One allocation, one scan.
    pub(super) fn replace_patterns(input: &str, matchers: &[Matcher]) -> String {
        let bytes = input.as_bytes();
        let len = bytes.len();
        let mut result = String::with_capacity(len);
        let mut i = 0;

        while i < len {
            if bytes[i].is_ascii() {
                let mut matched = false;
                for (is_start, consume) in matchers {
                    if is_start(bytes, i) {
                        if let Some((end, replacement)) = consume(bytes, i) {
                            result.push_str(replacement);
                            i = end;
                            matched = true;
                            break;
                        }
                    }
                }
                if matched {
                    continue;
                }
            }
            // Advance by full UTF-8 character to preserve multibyte sequences.
            // All pattern start bytes are ASCII, so non-ASCII bytes always land here.
            let ch_len = utf8_char_width(bytes[i]);
            result.push_str(&input[i..i + ch_len]);
            i += ch_len;
        }

        result
    }

    /// Single-pattern scanner used by individual normalizer functions.
    pub(super) fn replace_pattern<P, C>(input: &str, is_start: P, consume: C) -> String
    where
        P: Fn(&[u8], usize) -> bool,
        C: Fn(&[u8], usize) -> Option<(usize, &'static str)>,
    {
        let bytes = input.as_bytes();
        let len = bytes.len();
        let mut result = String::with_capacity(len);
        let mut i = 0;

        while i < len {
            if bytes[i].is_ascii() && is_start(bytes, i) {
                if let Some((end, replacement)) = consume(bytes, i) {
                    result.push_str(replacement);
                    i = end;
                    continue;
                }
            }
            let ch_len = utf8_char_width(bytes[i]);
            result.push_str(&input[i..i + ch_len]);
            i += ch_len;
        }

        result
    }

    fn utf8_char_width(b: u8) -> usize {
        match b {
            0..=0x7F => 1,
            0xC0..=0xDF => 2,
            0xE0..=0xEF => 3,
            0xF0..=0xF7 => 4,
            _ => 1,
        }
    }
}

/// One submodule per pattern. Each exposes `is_start(bytes, i) -> bool` and
/// `consume(bytes, i) -> Option<(end, replacement)>` so the engine can drive
/// them through a uniform [`engine::Matcher`] tuple.
mod patterns {
    fn is_hex_digit(b: u8) -> bool {
        b.is_ascii_digit() || (b'a'..=b'f').contains(&b) || (b'A'..=b'F').contains(&b)
    }

    pub mod hex_addresses {
        use super::is_hex_digit;

        pub fn is_start(bytes: &[u8], i: usize) -> bool {
            i + 2 < bytes.len()
                && bytes[i] == b'0'
                && bytes[i + 1] == b'x'
                && is_hex_digit(bytes[i + 2])
        }

        pub fn consume(bytes: &[u8], i: usize) -> Option<(usize, &'static str)> {
            let mut end = i + 2;
            while end < bytes.len() && is_hex_digit(bytes[end]) {
                end += 1;
            }
            let hex_len = end - i - 2;
            if hex_len >= 4 {
                Some((end, "0x_"))
            } else {
                None
            }
        }
    }

    pub mod hex_tokens {
        use super::is_hex_digit;

        pub fn is_start(bytes: &[u8], i: usize) -> bool {
            if !is_hex_digit(bytes[i]) {
                return false;
            }
            if i > 0 {
                let prev = bytes[i - 1];
                // Skip mid-run / inside-word positions — the start of the run
                // already had its chance, and we don't want to match inside
                // identifiers like `0x...` or `name_abcd1234`.
                if prev.is_ascii_alphabetic() || prev == b'_' || is_hex_digit(prev) {
                    return false;
                }
            }
            true
        }

        pub fn consume(bytes: &[u8], i: usize) -> Option<(usize, &'static str)> {
            let mut end = i;
            let mut has_alpha_hex = false;
            while end < bytes.len() && is_hex_digit(bytes[end]) {
                if bytes[end].is_ascii_alphabetic() {
                    has_alpha_hex = true;
                }
                end += 1;
            }
            if end < bytes.len() {
                let next = bytes[end];
                if next.is_ascii_alphabetic() || next == b'_' {
                    return None;
                }
            }
            let len = end - i;
            // Require 8+ chars and at least one alpha hex digit to avoid matching plain numbers.
            if len >= 8 && has_alpha_hex {
                Some((end, "_hex_"))
            } else {
                None
            }
        }
    }

    pub mod numeric_ids {
        pub fn is_start(bytes: &[u8], i: usize) -> bool {
            bytes[i].is_ascii_digit()
                && (i == 0 || !bytes[i - 1].is_ascii_alphanumeric() && bytes[i - 1] != b'_')
        }

        pub fn consume(bytes: &[u8], i: usize) -> Option<(usize, &'static str)> {
            let mut end = i;
            while end < bytes.len() && bytes[end].is_ascii_digit() {
                end += 1;
            }
            // Don't replace digits that are part of an alphanumeric word.
            if end < bytes.len() && (bytes[end].is_ascii_alphabetic() || bytes[end] == b'_') {
                return None;
            }
            if end > i {
                Some((end, "0"))
            } else {
                None
            }
        }
    }

    pub mod timestamps {
        pub fn is_start(bytes: &[u8], i: usize) -> bool {
            i + 6 < bytes.len()
                && bytes[i] == b'"'
                && bytes[i + 1].is_ascii_digit()
                && bytes[i + 2].is_ascii_digit()
                && bytes[i + 3].is_ascii_digit()
                && bytes[i + 4].is_ascii_digit()
                && bytes[i + 5] == b'-'
        }

        pub fn consume(bytes: &[u8], i: usize) -> Option<(usize, &'static str)> {
            let mut end = i + 1;
            while end < bytes.len() && bytes[end] != b'"' {
                end += 1;
            }
            if end < bytes.len() {
                end += 1; // include closing quote
                Some((end, "\"_timestamp_\""))
            } else {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_addresses_replaces_0x_prefixed() {
        assert_eq!(
            hex_addresses("object at 0x00007f3a1b2c3d48 done"),
            "object at 0x_ done"
        );
    }

    #[test]
    fn hex_addresses_ignores_short() {
        assert_eq!(hex_addresses("code 0x1F is fine"), "code 0x1F is fine");
    }

    #[test]
    fn numeric_ids_replaces_standalone_numbers() {
        assert_eq!(
            numeric_ids("id: 12345, name: \"hello\""),
            "id: 0, name: \"hello\""
        );
    }

    #[test]
    fn numeric_ids_preserves_embedded_digits() {
        assert_eq!(numeric_ids("SKU-12345 and v2"), "SKU-0 and v2");
    }

    #[test]
    fn timestamps_replaces_quoted_timestamps() {
        assert_eq!(
            timestamps("created_at: \"2026-03-25 03:56:10.370950000 +0000\", done"),
            "created_at: \"_timestamp_\", done"
        );
    }

    #[test]
    fn hex_tokens_replaces_long_bare_hex() {
        assert_eq!(
            hex_tokens("token: 3966ad2c7d6a51b5 end"),
            "token: _hex_ end"
        );
    }

    #[test]
    fn hex_tokens_preserves_non_hex_alpha() {
        // 'g' and 'h' are not hex digits, so "abcdefgh" is not a hex token
        assert_eq!(hex_tokens("word abcdefgh end"), "word abcdefgh end");
    }

    #[test]
    fn hex_tokens_matches_real_hex_token() {
        // "1a2b3c4d5e6f7a8b" is 16 hex chars with alpha — matches
        assert_eq!(hex_tokens("token 1a2b3c4d5e6f7a8b end"), "token _hex_ end");
    }

    #[test]
    fn hex_tokens_preserves_words_with_trailing_alpha() {
        // "abcdef01xyz" has non-hex trailing chars, so it's a word, not a token
        assert_eq!(hex_tokens("word abcdef01xyz end"), "word abcdef01xyz end");
    }

    #[test]
    fn hex_tokens_ignores_short_hex() {
        assert_eq!(hex_tokens("short abc123 end"), "short abc123 end");
    }

    #[test]
    fn hex_tokens_does_not_match_inside_0x_address() {
        assert_eq!(
            hex_tokens("addr 0x00007f3a1b2c3d48 end"),
            "addr 0x00007f3a1b2c3d48 end"
        );
    }

    #[test]
    fn compose_chains_normalizers() {
        let norm = compose(hex_addresses, numeric_ids);
        assert_eq!(norm("0x00ff at id: 999"), "0x_ at id: 0");
    }

    #[test]
    fn default_normalizer_handles_structured_error() {
        let norm = default_normalizer();
        let input = "undefined method 'render_widget' for #<App::Component::Widget:0x00007f3a1b2c3d48 @parent_id=1234, @record=#<App::Model id: 567, created_at: \"2026-03-25 03:56:10.370950000 +0000\">>";
        let result = norm(input);
        assert_eq!(
            result,
            "undefined method 'render_widget' for #<App::Component::Widget:0x_ @parent_id=0, @record=#<App::Model id: 0, created_at: \"_timestamp_\">>"
        );
    }

    #[test]
    fn default_normalizer_is_identity_on_fqcn() {
        let norm = default_normalizer();
        let input = "org.blogengine.api.domain.PostMapper.findById";
        assert_eq!(norm(input), input);
    }

    use quickcheck_macros::quickcheck;

    #[quickcheck]
    fn prop_identity_preserves_input(s: String) -> bool {
        identity(&s) == s
    }

    #[quickcheck]
    fn prop_hex_addresses_idempotent(s: String) -> bool {
        let once = hex_addresses(&s);
        let twice = hex_addresses(&once);
        once == twice
    }

    #[quickcheck]
    fn prop_numeric_ids_idempotent(s: String) -> bool {
        let once = numeric_ids(&s);
        let twice = numeric_ids(&once);
        once == twice
    }

    #[quickcheck]
    fn prop_timestamps_idempotent(s: String) -> bool {
        let once = timestamps(&s);
        let twice = timestamps(&once);
        once == twice
    }

    #[quickcheck]
    fn prop_hex_tokens_idempotent(s: String) -> bool {
        let once = hex_tokens(&s);
        let twice = hex_tokens(&once);
        once == twice
    }

    #[quickcheck]
    fn prop_default_normalizer_idempotent(s: String) -> bool {
        let norm = default_normalizer();
        let once = norm(&s);
        let twice = norm(&once);
        once == twice
    }

    #[quickcheck]
    fn prop_normalizers_produce_valid_utf8(s: String) -> bool {
        // Each normalizer should return a valid String (enforced by type system),
        // but verify no panics on arbitrary input including multibyte UTF-8.
        let _ = hex_addresses(&s);
        let _ = numeric_ids(&s);
        let _ = timestamps(&s);
        let _ = hex_tokens(&s);
        let _ = default_normalizer()(&s);
        true
    }

    #[quickcheck]
    fn prop_compose_matches_sequential_application(s: String) -> bool {
        let composed = compose(hex_addresses, numeric_ids);
        let sequential = numeric_ids(&hex_addresses(&s));
        composed(&s) == sequential
    }
}
