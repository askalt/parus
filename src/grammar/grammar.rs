use either::Either;
use std::{collections::VecDeque, hash::Hash};

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
#[derive(Clone)]
pub struct Epsilon {}

/// Describes a context-free grammar.
pub trait Grammar<S>
where
    S: Symbol,
{
    /// Get productions for the specific symbol.
    /// Will be called for only non-terminal symbols.
    fn get_productions(&self, symbol: &S) -> &[Either<&[S], Epsilon>];

    /// Makes an iterator from the grammar.
    fn into_iterator(self) -> GrammarIterator<S, Self>
    where
        Self: Sized,
    {
        return GrammarIterator::new(self);
    }
}

/// Visits the specified grammar, using BFS.
/// If the grammar produces the same string several times, returns it several times.
pub struct GrammarIterator<S: Symbol, G: Grammar<S>> {
    grammar: G,
    queue: VecDeque<Vec<S>>,
}

impl<S: Symbol, G: Grammar<S>> GrammarIterator<S, G> {
    /// Creates new `GrammarIterator` over some grammar.
    fn new(grammar: G) -> Self {
        Self {
            grammar: grammar,
            queue: VecDeque::from([vec![S::start_non_terminal()]]),
        }
    }
}

impl<S: Symbol, G: Grammar<S>> Iterator for GrammarIterator<S, G> {
    type Item = Vec<S>;

    /// Return next derivate string.
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(str) = self.queue.pop_front() {
            // Rewrite first non-terminal.
            let non_terminal = str.iter().enumerate().find(|(_, e)| e.is_non_terminal());
            if non_terminal.is_none() {
                // Strings does not contain non-terminal => it in language.
                return Some(str);
            }
            let (i, s) = non_terminal.unwrap();
            for production in self.grammar.get_productions(s) {
                let mut nxt = str.clone();
                if production.is_right() {
                    // We found epsilon production, so remove this symbol.
                    nxt.remove(i);
                } else {
                    // We found non-epsilon production, so replace subarray.
                    let production = production.clone().left().unwrap();
                    nxt.splice(i..i + 1, production.iter().cloned());
                }
                self.queue.push_back(nxt);
            }
        }
        None
    }
}
