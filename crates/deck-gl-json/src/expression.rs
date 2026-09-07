//! Expression strings for `@@=` accessors.
//!
//! Mirrors the jsep grammar used by `@deck.gl/json`: literals, identifiers, member access, array
//! literals, unary, binary and conditional operators. Function calls are rejected, as in deck.gl.
//! Expressions evaluate against one JSON row with JavaScript-like coercion rules, so a spec written
//! for pydeck or the deck.gl playground behaves the same here.

use std::fmt;

use serde_json::{Number, Value};

/// A parse error with the offending expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpressionError(pub String);

impl fmt::Display for ExpressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ExpressionError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Negate,
    Plus,
    Not,
    BitNot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Or,
    And,
    BitOr,
    BitXor,
    BitAnd,
    Eq,
    NotEq,
    StrictEq,
    StrictNotEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    Shl,
    Shr,
    UShr,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

impl BinaryOp {
    /// Operator and jsep precedence for a punctuation token.
    fn from_token(token: &str) -> Option<(BinaryOp, u8)> {
        use BinaryOp::*;
        Some(match token {
            "||" => (Or, 1),
            "&&" => (And, 2),
            "|" => (BitOr, 3),
            "^" => (BitXor, 4),
            "&" => (BitAnd, 5),
            "==" => (Eq, 6),
            "!=" => (NotEq, 6),
            "===" => (StrictEq, 6),
            "!==" => (StrictNotEq, 6),
            "<" => (Lt, 7),
            ">" => (Gt, 7),
            "<=" => (LtEq, 7),
            ">=" => (GtEq, 7),
            "<<" => (Shl, 8),
            ">>" => (Shr, 8),
            ">>>" => (UShr, 8),
            "+" => (Add, 9),
            "-" => (Sub, 9),
            "*" => (Mul, 10),
            "/" => (Div, 10),
            "%" => (Rem, 10),
            _ => return None,
        })
    }
}

/// A parsed accessor expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Literal(Value),
    /// The row itself (`this`, or the deck.gl shorthand `-`).
    Row,
    Identifier(String),
    Member {
        object: Box<Expr>,
        property: Box<Expr>,
    },
    Array(Vec<Expr>),
    Unary {
        op: UnaryOp,
        argument: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Conditional {
        test: Box<Expr>,
        consequent: Box<Expr>,
        alternate: Box<Expr>,
    },
}

impl Expr {
    /// Parse an accessor expression. The bare string `-` is deck.gl shorthand for the row itself.
    pub fn parse(source: &str) -> Result<Expr, ExpressionError> {
        if source.trim() == "-" {
            return Ok(Expr::Row);
        }
        let tokens = tokenize(source)?;
        let mut parser = Parser {
            tokens,
            pos: 0,
            source,
        };
        let expr = parser.expression()?;
        if parser.pos < parser.tokens.len() {
            return Err(ExpressionError(format!(
                "unexpected {} after the end of expression `{source}`",
                parser.tokens[parser.pos]
            )));
        }
        Ok(expr)
    }

    /// True when the expression never reads the row, so it can be evaluated once.
    pub fn is_constant(&self) -> bool {
        match self {
            Expr::Literal(_) => true,
            Expr::Row | Expr::Identifier(_) => false,
            Expr::Member { object, property } => object.is_constant() && property.is_constant(),
            Expr::Array(items) => items.iter().all(Expr::is_constant),
            Expr::Unary { argument, .. } => argument.is_constant(),
            Expr::Binary { left, right, .. } => left.is_constant() && right.is_constant(),
            Expr::Conditional {
                test,
                consequent,
                alternate,
            } => test.is_constant() && consequent.is_constant() && alternate.is_constant(),
        }
    }

    /// Evaluate against a row. Missing fields evaluate to null instead of failing.
    pub fn eval(&self, row: &Value) -> Value {
        match self {
            Expr::Literal(value) => value.clone(),
            Expr::Row => row.clone(),
            Expr::Identifier(name) => row.get(name).cloned().unwrap_or(Value::Null),
            Expr::Member { object, property } => member(&object.eval(row), &property.eval(row)),
            Expr::Array(items) => Value::Array(items.iter().map(|item| item.eval(row)).collect()),
            Expr::Unary { op, argument } => {
                let value = argument.eval(row);
                match op {
                    UnaryOp::Negate => number(-to_number(&value)),
                    UnaryOp::Plus => number(to_number(&value)),
                    UnaryOp::Not => Value::Bool(!truthy(&value)),
                    UnaryOp::BitNot => number(!to_int32(&value) as f64),
                }
            }
            Expr::Binary { op, left, right } => match op {
                BinaryOp::Or => {
                    let l = left.eval(row);
                    if truthy(&l) {
                        l
                    } else {
                        right.eval(row)
                    }
                }
                BinaryOp::And => {
                    let l = left.eval(row);
                    if truthy(&l) {
                        right.eval(row)
                    } else {
                        l
                    }
                }
                _ => binary(*op, &left.eval(row), &right.eval(row)),
            },
            Expr::Conditional {
                test,
                consequent,
                alternate,
            } => {
                if truthy(&test.eval(row)) {
                    consequent.eval(row)
                } else {
                    alternate.eval(row)
                }
            }
        }
    }
}

/// Integral results become JSON integers, so `2 * 3` prints as `6` rather than `6.0`.
fn number(value: f64) -> Value {
    if value.fract() == 0.0 && value.abs() < 9.0e15 {
        Value::from(value as i64)
    } else {
        Number::from_f64(value).map(Value::Number).unwrap_or(Value::Null)
    }
}

/// JavaScript `ToNumber`.
pub fn to_number(value: &Value) -> f64 {
    match value {
        Value::Null => 0.0,
        Value::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        Value::Number(n) => n.as_f64().unwrap_or(f64::NAN),
        Value::String(s) => {
            let s = s.trim();
            if s.is_empty() {
                0.0
            } else {
                s.parse().unwrap_or(f64::NAN)
            }
        }
        Value::Array(items) => match items.as_slice() {
            [] => 0.0,
            [single] => to_number(single),
            _ => f64::NAN,
        },
        Value::Object(_) => f64::NAN,
    }
}

fn to_int32(value: &Value) -> i32 {
    let n = to_number(value);
    if !n.is_finite() {
        return 0;
    }
    n.trunc().rem_euclid(4_294_967_296.0) as u32 as i32
}

/// JavaScript truthiness.
pub fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0 && !f.is_nan()).unwrap_or(false),
        Value::String(s) => !s.is_empty(),
        Value::Array(_) | Value::Object(_) => true,
    }
}

/// JavaScript `ToString`, close enough for concatenation in accessors.
pub fn to_display_string(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => match n.as_f64() {
            Some(f) if f.fract() == 0.0 && f.abs() < 1e21 => format!("{}", f as i64),
            Some(f) => f.to_string(),
            None => n.to_string(),
        },
        Value::String(s) => s.clone(),
        Value::Array(items) => items.iter().map(to_display_string).collect::<Vec<_>>().join(","),
        Value::Object(_) => "[object Object]".to_string(),
    }
}

fn member(object: &Value, key: &Value) -> Value {
    match (object, key) {
        (Value::Array(items), Value::Number(n)) => n
            .as_f64()
            .filter(|i| *i >= 0.0 && i.fract() == 0.0)
            .and_then(|i| items.get(i as usize))
            .cloned()
            .unwrap_or(Value::Null),
        (Value::Array(items), Value::String(s)) => {
            if s == "length" {
                number(items.len() as f64)
            } else {
                s.parse::<usize>()
                    .ok()
                    .and_then(|i| items.get(i))
                    .cloned()
                    .unwrap_or(Value::Null)
            }
        }
        (Value::Object(map), Value::String(s)) => map.get(s).cloned().unwrap_or(Value::Null),
        (Value::Object(map), Value::Number(n)) => map
            .get(&to_display_string(&Value::Number(n.clone())))
            .cloned()
            .unwrap_or(Value::Null),
        (Value::String(s), Value::String(k)) if k == "length" => number(s.chars().count() as f64),
        _ => Value::Null,
    }
}

fn loose_eq(l: &Value, r: &Value) -> bool {
    match (l, r) {
        (Value::Null, Value::Null) => true,
        (Value::Null, _) | (_, Value::Null) => false,
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Array(_), Value::Array(_)) | (Value::Object(_), Value::Object(_)) => l == r,
        _ => to_number(l) == to_number(r),
    }
}

fn strict_eq(l: &Value, r: &Value) -> bool {
    match (l, r) {
        (Value::Number(a), Value::Number(b)) => a.as_f64() == b.as_f64(),
        (Value::Null, Value::Null) => true,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Array(_), Value::Array(_)) | (Value::Object(_), Value::Object(_)) => l == r,
        _ => false,
    }
}

fn compare(l: &Value, r: &Value, op: BinaryOp) -> bool {
    if let (Value::String(a), Value::String(b)) = (l, r) {
        return match op {
            BinaryOp::Lt => a < b,
            BinaryOp::Gt => a > b,
            BinaryOp::LtEq => a <= b,
            _ => a >= b,
        };
    }
    let (a, b) = (to_number(l), to_number(r));
    match op {
        BinaryOp::Lt => a < b,
        BinaryOp::Gt => a > b,
        BinaryOp::LtEq => a <= b,
        _ => a >= b,
    }
}

fn binary(op: BinaryOp, l: &Value, r: &Value) -> Value {
    use BinaryOp::*;
    match op {
        Add => {
            if l.is_string() || r.is_string() {
                Value::String(format!("{}{}", to_display_string(l), to_display_string(r)))
            } else {
                number(to_number(l) + to_number(r))
            }
        }
        Sub => number(to_number(l) - to_number(r)),
        Mul => number(to_number(l) * to_number(r)),
        Div => number(to_number(l) / to_number(r)),
        Rem => number(to_number(l) % to_number(r)),
        Eq => Value::Bool(loose_eq(l, r)),
        NotEq => Value::Bool(!loose_eq(l, r)),
        StrictEq => Value::Bool(strict_eq(l, r)),
        StrictNotEq => Value::Bool(!strict_eq(l, r)),
        Lt | Gt | LtEq | GtEq => Value::Bool(compare(l, r, op)),
        BitOr => number((to_int32(l) | to_int32(r)) as f64),
        BitXor => number((to_int32(l) ^ to_int32(r)) as f64),
        BitAnd => number((to_int32(l) & to_int32(r)) as f64),
        Shl => number(to_int32(l).wrapping_shl(to_int32(r) as u32 & 31) as f64),
        Shr => number((to_int32(l) >> (to_int32(r) as u32 & 31)) as f64),
        UShr => number(((to_int32(l) as u32) >> (to_int32(r) as u32 & 31)) as f64),
        // Short circuit operators are evaluated by Expr::eval; treat a stray one as false
        Or | And => Value::Bool(false),
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Str(String),
    Ident(String),
    Punct(&'static str),
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Number(n) => write!(f, "number {n}"),
            Token::Str(s) => write!(f, "string {s:?}"),
            Token::Ident(s) => write!(f, "identifier `{s}`"),
            Token::Punct(p) => write!(f, "`{p}`"),
        }
    }
}

/// Longest tokens first so the matcher is greedy.
const PUNCTUATION: [&str; 31] = [
    "===", "!==", ">>>", "==", "!=", "<=", ">=", "&&", "||", "<<", ">>", "+", "-", "*", "/", "%", "<", ">",
    "&", "|", "^", "!", "~", "?", ":", ".", ",", "(", ")", "[", "]",
];

fn is_ident_start(c: char) -> bool {
    c.is_alphabetic() || c == '_' || c == '$'
}

fn is_ident_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '$'
}

fn tokenize(source: &str) -> Result<Vec<Token>, ExpressionError> {
    let chars: Vec<char> = source.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() || (c == '.' && chars.get(i + 1).is_some_and(char::is_ascii_digit)) {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                i += 1;
                if i < chars.len() && (chars[i] == '+' || chars[i] == '-') {
                    i += 1;
                }
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let text: String = chars[start..i].iter().collect();
            let value = text
                .parse::<f64>()
                .map_err(|_| ExpressionError(format!("invalid number `{text}` in `{source}`")))?;
            tokens.push(Token::Number(value));
        } else if c == '\'' || c == '"' {
            let quote = c;
            i += 1;
            let mut text = String::new();
            let mut closed = false;
            while i < chars.len() {
                let ch = chars[i];
                i += 1;
                if ch == quote {
                    closed = true;
                    break;
                }
                if ch == '\\' && i < chars.len() {
                    let escaped = chars[i];
                    i += 1;
                    text.push(match escaped {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        other => other,
                    });
                } else {
                    text.push(ch);
                }
            }
            if !closed {
                return Err(ExpressionError(format!("unterminated string in `{source}`")));
            }
            tokens.push(Token::Str(text));
        } else if is_ident_start(c) {
            let start = i;
            while i < chars.len() && is_ident_char(chars[i]) {
                i += 1;
            }
            tokens.push(Token::Ident(chars[start..i].iter().collect()));
        } else {
            let rest: String = chars[i..chars.len().min(i + 3)].iter().collect();
            let punct = PUNCTUATION
                .iter()
                .find(|p| rest.starts_with(**p))
                .ok_or_else(|| ExpressionError(format!("unexpected character `{c}` in `{source}`")))?;
            tokens.push(Token::Punct(punct));
            i += punct.len();
        }
    }
    Ok(tokens)
}

struct Parser<'a> {
    tokens: Vec<Token>,
    pos: usize,
    source: &'a str,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.pos).cloned();
        self.pos += 1;
        token
    }

    fn eat(&mut self, punct: &str) -> bool {
        if matches!(self.peek(), Some(Token::Punct(p)) if *p == punct) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, punct: &str) -> Result<(), ExpressionError> {
        if self.eat(punct) {
            Ok(())
        } else {
            Err(ExpressionError(match self.peek() {
                Some(token) => format!("expected `{punct}` but found {token} in `{}`", self.source),
                None => format!("expected `{punct}` at the end of `{}`", self.source),
            }))
        }
    }

    fn expression(&mut self) -> Result<Expr, ExpressionError> {
        let test = self.binary(0)?;
        if self.eat("?") {
            let consequent = self.expression()?;
            self.expect(":")?;
            let alternate = self.expression()?;
            return Ok(Expr::Conditional {
                test: Box::new(test),
                consequent: Box::new(consequent),
                alternate: Box::new(alternate),
            });
        }
        Ok(test)
    }

    fn binary(&mut self, min_precedence: u8) -> Result<Expr, ExpressionError> {
        let mut left = self.unary()?;
        while let Some(Token::Punct(punct)) = self.peek() {
            let Some((op, precedence)) = BinaryOp::from_token(punct) else {
                break;
            };
            if precedence <= min_precedence {
                break;
            }
            self.pos += 1;
            let right = self.binary(precedence)?;
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn unary(&mut self) -> Result<Expr, ExpressionError> {
        if let Some(Token::Punct(punct)) = self.peek() {
            let op = match *punct {
                "-" => Some(UnaryOp::Negate),
                "+" => Some(UnaryOp::Plus),
                "!" => Some(UnaryOp::Not),
                "~" => Some(UnaryOp::BitNot),
                _ => None,
            };
            if let Some(op) = op {
                self.pos += 1;
                let argument = self.unary()?;
                return Ok(Expr::Unary {
                    op,
                    argument: Box::new(argument),
                });
            }
        }
        self.postfix()
    }

    fn postfix(&mut self) -> Result<Expr, ExpressionError> {
        let mut expr = self.primary()?;
        loop {
            if self.eat(".") {
                match self.next() {
                    Some(Token::Ident(name)) => {
                        expr = Expr::Member {
                            object: Box::new(expr),
                            property: Box::new(Expr::Literal(Value::String(name))),
                        }
                    }
                    Some(token) => {
                        return Err(ExpressionError(format!(
                            "expected a property name after `.` but found {token} in `{}`",
                            self.source
                        )))
                    }
                    None => {
                        return Err(ExpressionError(format!(
                            "expected a property name after `.` at the end of `{}`",
                            self.source
                        )))
                    }
                }
            } else if self.eat("[") {
                let property = self.expression()?;
                self.expect("]")?;
                expr = Expr::Member {
                    object: Box::new(expr),
                    property: Box::new(property),
                };
            } else if matches!(self.peek(), Some(Token::Punct("("))) {
                return Err(ExpressionError(format!(
                    "function calls are not allowed in JSON expressions: `{}`",
                    self.source
                )));
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn primary(&mut self) -> Result<Expr, ExpressionError> {
        match self.next() {
            Some(Token::Number(n)) => Ok(Expr::Literal(number(n))),
            Some(Token::Str(s)) => Ok(Expr::Literal(Value::String(s))),
            Some(Token::Ident(name)) => Ok(match name.as_str() {
                "true" => Expr::Literal(Value::Bool(true)),
                "false" => Expr::Literal(Value::Bool(false)),
                "null" | "undefined" => Expr::Literal(Value::Null),
                "this" => Expr::Row,
                _ => Expr::Identifier(name),
            }),
            Some(Token::Punct("(")) => {
                let expr = self.expression()?;
                self.expect(")")?;
                Ok(expr)
            }
            Some(Token::Punct("[")) => {
                let mut items = Vec::new();
                if !self.eat("]") {
                    loop {
                        items.push(self.expression()?);
                        if self.eat("]") {
                            break;
                        }
                        self.expect(",")?;
                    }
                }
                Ok(Expr::Array(items))
            }
            Some(token) => Err(ExpressionError(format!(
                "unexpected {token} in `{}`",
                self.source
            ))),
            None => Err(ExpressionError(format!(
                "unexpected end of expression `{}`",
                self.source
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn eval(source: &str, row: Value) -> Value {
        Expr::parse(source).unwrap().eval(&row)
    }

    #[test]
    fn reads_fields_and_nested_members() {
        let row = json!({"coordinates": [1.5, 2.5], "properties": {"valuePerSqm": 3.5, "name": "a"}});
        assert_eq!(eval("coordinates", row.clone()), json!([1.5, 2.5]));
        assert_eq!(eval("properties.valuePerSqm", row.clone()), json!(3.5));
        assert_eq!(eval("properties['name']", row.clone()), json!("a"));
        assert_eq!(eval("coordinates[1]", row.clone()), json!(2.5));
        assert_eq!(eval("coordinates.length", row.clone()), json!(2));
        assert_eq!(eval("missing.deeper", row.clone()), Value::Null);
        assert_eq!(eval("this.coordinates[0]", row.clone()), json!(1.5));
        assert_eq!(eval("-", row.clone()), row);
    }

    #[test]
    fn builds_arrays_from_fields() {
        let row = json!({"lng": -122.4, "lat": 37.8, "r": 10, "g": 20, "b": 30});
        assert_eq!(eval("[lng, lat]", row.clone()), json!([-122.4, 37.8]));
        assert_eq!(eval("[r, g, b, 255]", row), json!([10, 20, 30, 255]));
        assert_eq!(eval("[]", Value::Null), json!([]));
    }

    #[test]
    fn applies_precedence_and_associativity() {
        assert_eq!(eval("1 + 2 * 3", Value::Null), json!(7));
        assert_eq!(eval("(1 + 2) * 3", Value::Null), json!(9));
        assert_eq!(eval("10 - 2 - 3", Value::Null), json!(5));
        assert_eq!(eval("2 * -3", Value::Null), json!(-6));
        assert_eq!(eval("7 % 4", Value::Null), json!(3));
        assert_eq!(eval("1 + 2 > 2 && 3 < 4", Value::Null), json!(true));
        assert_eq!(eval("5 >> 1 | 8", Value::Null), json!(10));
    }

    #[test]
    fn conditionals_and_logic_follow_javascript() {
        let row = json!({"value": 5, "flag": false, "name": null});
        assert_eq!(eval("value > 3 ? 'hi' : 'lo'", row.clone()), json!("hi"));
        assert_eq!(eval("value > 30 ? 'hi' : 'lo'", row.clone()), json!("lo"));
        assert_eq!(eval("name || 'default'", row.clone()), json!("default"));
        assert_eq!(eval("value && flag", row.clone()), json!(false));
        assert_eq!(eval("!flag", row.clone()), json!(true));
        assert_eq!(eval("value == '5'", row.clone()), json!(true));
        assert_eq!(eval("value === '5'", row.clone()), json!(false));
        assert_eq!(eval("value !== 5", row), json!(false));
    }

    #[test]
    fn concatenates_strings() {
        let row = json!({"id": 7, "kind": "pin"});
        assert_eq!(eval("'icon_' + id", row.clone()), json!("icon_7"));
        assert_eq!(eval("kind + \"-\" + (id * 2)", row), json!("pin-14"));
    }

    #[test]
    fn rejects_calls_and_garbage() {
        assert!(Expr::parse("Math.max(a, b)")
            .unwrap_err()
            .0
            .contains("function calls"));
        assert!(Expr::parse("a b").is_err());
        assert!(Expr::parse("a +").is_err());
        assert!(Expr::parse("'open").is_err());
        assert!(Expr::parse("a # b").is_err());
    }

    #[test]
    fn detects_constant_expressions() {
        assert!(Expr::parse("[1, 2, 3]").unwrap().is_constant());
        assert!(Expr::parse("2 * 3 > 1 ? 4 : 5").unwrap().is_constant());
        assert!(!Expr::parse("[lng, lat]").unwrap().is_constant());
        assert!(!Expr::parse("-").unwrap().is_constant());
    }
}
