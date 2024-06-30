use std::hash::Hash;

use crate::either::Either;

/// Describes the symbols in the specific grammar.
pub trait Symbol: Eq + Hash + Ord + Clone {
    /// Returns true is symbol is a terminal.
    fn is_terminal(&self) -> bool;

    /// Returns true is symbol is a non-terminal.
    fn is_non_terminal(&self) -> bool {
        return !self.is_terminal();
    }

    /// Returns a start non terminal for the grammar.
    fn start_non_terminal() -> Self;

    /// Tries to compare and accept the actual data for a terminal.
    /// Returns true in the case of success.
    ///
    /// For example, we are trying to parse the terminal `Int`,
    /// in production this symbol is some empty `Int` structure,
    /// lexer returns `Int(12345)` and next we try to fill the data with actual value 12345.
    /// In this situation `Int`.is_accept(`Int`(12345)) must return true.
    ///
    /// Or, lexer can return some other symbol, e.g., `Float`, and in this situation,
    /// is_accept(...) must return false.
    fn is_accept(&self, oth: &Self) -> bool;
}

/// Equivalent of an empty string.
pub struct Epsilon {}

/// Describes a context-free grammar.
pub trait Grammar<S>
where
    S: Symbol,
{
    /// Get productions for the specific symbol.
    /// Will be called for only non-terminal symbols.
    fn get_productions(&self, symbol: &S) -> &[Either<&[S], Epsilon>];
}

/// Describes a lexer over some grammar.
pub trait Lexer<S>
where
    S: Symbol,
{
    /// Returns next symbol, if one exists.
    /// Else returns `None`.
    fn next(&mut self) -> Option<S> {
        let ret = self.cur();
        if matches!(ret, Some(_)) {
            self.shift();
        }
        ret
    }

    /// Returns current symbol, if one exists.
    /// Else returns `None`.
    fn cur(&self) -> Option<S>;

    /// Shifts current position by 1 to the right.
    fn shift(&mut self);
}
