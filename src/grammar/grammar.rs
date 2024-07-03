use either::Either;
use rand::Rng;
use std::{collections::VecDeque, hash::Hash, marker::PhantomData};

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

/// Applies production to the string and returns the count of terminals.
fn apply_production<S: Symbol>(
    str: &mut Vec<S>,
    i: usize,
    production: &Either<&[S], Epsilon>,
) -> i32 {
    if production.is_right() {
        // Epsilon production, so remove this symbol.
        str.remove(i);
        0
    } else {
        // Non-epsilon production, so replace subarray.
        let production = production.clone().left().unwrap();
        let mut count_non_terminals = 0;
        str.splice(
            i..i + 1,
            production
                .iter()
                .map(|it| {
                    if it.is_terminal() {
                        count_non_terminals += 1;
                    }
                    it
                })
                .cloned(),
        );
        count_non_terminals
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
                apply_production(&mut nxt, i, production);
                self.queue.push_back(nxt);
            }
        }
        None
    }
}

pub struct RandomGrammarIterator<S: Symbol, G: Grammar<S>> {
    grammar: G,
    /// Tries not choose productions with only terminals while length < min_length.
    min_lentgth: usize,
    /// If exceeds max_length, tries to choose production without non-terminals.
    max_length: usize,
    phantom: PhantomData<S>,
    rng: rand::rngs::ThreadRng,
}

impl<S: Symbol, G: Grammar<S>> RandomGrammarIterator<S, G> {
    pub fn new(grammar: G, min_length: usize, max_length: usize) -> Self {
        Self {
            grammar: grammar,
            min_lentgth: min_length,
            max_length: max_length,
            phantom: PhantomData,
            rng: rand::thread_rng(),
        }
    }
}

/// Returns a random index preferring `important` indices set.
fn take_random_index(
    important: &Vec<usize>,
    non_important: &Vec<usize>,
    rng: &mut rand::rngs::ThreadRng,
) -> usize {
    assert!(!important.is_empty() || !non_important.is_empty());
    let id = rng.gen::<usize>();
    if important.is_empty() {
        non_important[id % non_important.len()]
    } else {
        important[id % important.len()]
    }
}

/// Returns random strings over grammar. Never return None.
impl<S: Symbol, G: Grammar<S>> Iterator for RandomGrammarIterator<S, G> {
    type Item = Vec<S>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut cur = vec![S::start_non_terminal()];
        // Current length in non-terminals.
        let mut length = 0;

        loop {
            let non_terminal_idx: Vec<_> = (0..cur.len())
                .filter(|i| cur[*i].is_non_terminal())
                .collect();
            if non_terminal_idx.is_empty() {
                assert!(cur.len() == length);
                return Some(cur);
            }
            let id = non_terminal_idx[self.rng.gen::<usize>() % non_terminal_idx.len()];
            let productions = self.grammar.get_productions(&cur[id]);
            assert!(!productions.is_empty());

            let prod_id = if length >= self.min_lentgth && length < self.max_length {
                // Choose a random production.
                let production_id = self.rng.gen::<usize>() % productions.len();
                production_id
            } else {
                let (with_non_terminals, without_non_terminals): (Vec<_>, Vec<_>) =
                    (0..productions.len()).partition(|i| {
                        if productions[*i].is_right() {
                            false
                        } else {
                            productions[*i]
                                .as_ref()
                                .left()
                                .unwrap()
                                .iter()
                                .any(|t| t.is_non_terminal())
                        }
                    });
                if length < self.min_lentgth {
                    // Try to thoose a production with non-terminals.
                    take_random_index(&with_non_terminals, &without_non_terminals, &mut self.rng)
                } else {
                    // length >= self.max_length
                    // Try to choose a production without non-terminals.
                    take_random_index(&without_non_terminals, &with_non_terminals, &mut self.rng)
                }
            };

            length += apply_production(&mut cur, id, &productions[prod_id]) as usize;
        }
    }
}
