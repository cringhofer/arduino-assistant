//! Persistent history in SQLite: every turn (what was heard, intent, actions, reply,
//! photo) plus the recorded command and reply audio of the most recent turns.

use std::sync::Mutex;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OptionalExtension, params};

use crate::status::Turn;

pub struct History {
    conn: Mutex<Connection>,
    /// Keep audio only for this many most recent turns (text is kept forever).
    keep_audio: u64,
}

/// Schema migrations, applied in order inside a transaction each. The database's
/// `PRAGMA user_version` is the number of migrations applied. Only ever append to this
/// list; never edit an entry that has been released.
const MIGRATIONS: &[&str] = &[
    // 1: turns (IF NOT EXISTS: databases from before migrations already have it)
    "CREATE TABLE IF NOT EXISTS turns (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        at_ms        INTEGER NOT NULL,
        trigger      TEXT NOT NULL,
        heard        TEXT,
        heard_engine TEXT,
        heard_alt    TEXT,
        alt_engine   TEXT,
        intent       TEXT,
        actions      TEXT NOT NULL DEFAULT '[]',
        said         TEXT,
        outcome      TEXT,
        duration_ms  INTEGER,
        photo        TEXT,
        heard_wav    BLOB,
        reply_wav    BLOB
    );
    CREATE INDEX IF NOT EXISTS turns_at ON turns(at_ms);",
    // 2: per-stage latency
    "CREATE TABLE IF NOT EXISTS latency (
        turn_id INTEGER,
        at_ms   INTEGER NOT NULL,
        stage   TEXT NOT NULL,
        ms      REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS latency_at ON latency(at_ms);
    CREATE INDEX IF NOT EXISTS latency_turn ON latency(turn_id);",
];

/// Brings the schema up to date. Refuses databases from a newer program version.
fn migrate(conn: &mut Connection, migrations: &[&str]) -> Result<()> {
    let version = conn.pragma_query_value(None, "user_version", |r| r.get::<_, i64>(0))? as usize;
    if version > migrations.len() {
        bail!("database schema v{version} is newer than this program supports (v{})", migrations.len());
    }
    for (i, sql) in migrations.iter().enumerate().skip(version) {
        let tx = conn.transaction()?;
        tx.execute_batch(sql).with_context(|| format!("schema migration {}", i + 1))?;
        tx.pragma_update(None, "user_version", (i + 1) as i64)?;
        tx.commit()?;
        log::info!("history: schema migrated to v{}", i + 1);
    }
    Ok(())
}

/// One latency measurement: how long a stage took (`turn` is None for periodic ones,
/// like the wake detector load).
#[derive(Debug, Clone, serde::Serialize)]
pub struct Sample {
    pub turn: Option<u64>,
    pub at: u64,
    pub stage: String,
    pub ms: f32,
}

impl History {
    pub fn open(path: &str, keep_audio: u64) -> Result<Self> {
        if let Some(dir) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(dir).ok();
        }
        let mut conn = Connection::open(path).with_context(|| format!("opening {path}"))?;
        // WAL: the status page can read while the assistant writes.
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        migrate(&mut conn, MIGRATIONS)?;
        Ok(Self { conn: Mutex::new(conn), keep_audio })
    }

    fn db(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Creates a turn and returns its id.
    pub fn insert(&self, at_ms: u64, trigger: &str) -> Result<u64> {
        let db = self.db();
        db.execute("INSERT INTO turns (at_ms, trigger) VALUES (?1, ?2)", params![at_ms as i64, trigger])?;
        Ok(db.last_insert_rowid() as u64)
    }

    /// Stores the text fields of a turn.
    pub fn save(&self, t: &Turn) -> Result<()> {
        self.db().execute(
            "UPDATE turns SET heard = ?2, heard_engine = ?3, heard_alt = ?4, alt_engine = ?5,
                 intent = ?6, actions = ?7, said = ?8, outcome = ?9, duration_ms = ?10, photo = ?11
             WHERE id = ?1",
            params![
                t.id as i64,
                t.heard,
                t.heard_engine,
                t.heard_alt,
                t.alt_engine,
                t.intent,
                serde_json::to_string(&t.actions)?,
                t.said,
                t.outcome,
                t.duration_ms.map(|d| d as i64),
                t.photo,
            ],
        )?;
        Ok(())
    }

    /// Stores the recorded command (`reply = false`) or reply audio as WAV and drops audio
    /// of turns older than the `keep_audio` most recent ones.
    pub fn set_audio(&self, id: u64, reply: bool, wav: &[u8]) -> Result<()> {
        let db = self.db();
        let col = if reply { "reply_wav" } else { "heard_wav" };
        db.execute(&format!("UPDATE turns SET {col} = ?2 WHERE id = ?1"), params![id as i64, wav])?;
        db.execute(
            "UPDATE turns SET heard_wav = NULL, reply_wav = NULL
             WHERE id <= (SELECT MAX(id) FROM turns) - ?1 AND (heard_wav IS NOT NULL OR reply_wav IS NOT NULL)",
            params![self.keep_audio as i64],
        )?;
        Ok(())
    }

    /// Size and contents of the database, for the Info page.
    pub fn stats(&self, path: &str) -> serde_json::Value {
        let db = self.db();
        let n = |sql: &str| db.query_row(sql, [], |r| r.get::<_, i64>(0)).ok();
        serde_json::json!({
            "path": path,
            "bytes": crate::info::size(path) + crate::info::size(&format!("{path}-wal")),
            "schema_version": n("PRAGMA user_version"),
            "schema_latest": MIGRATIONS.len(),
            "turns": n("SELECT COUNT(*) FROM turns"),
            "turns_with_audio": n("SELECT COUNT(*) FROM turns WHERE heard_wav IS NOT NULL OR reply_wav IS NOT NULL"),
            "latency_samples": n("SELECT COUNT(*) FROM latency"),
            "keep_audio": self.keep_audio,
        })
    }

    pub fn add_latency(&self, s: &Sample) -> Result<()> {
        self.db().execute(
            "INSERT INTO latency (turn_id, at_ms, stage, ms) VALUES (?1, ?2, ?3, ?4)",
            params![s.turn.map(|t| t as i64), s.at as i64, s.stage, s.ms as f64],
        )?;
        Ok(())
    }

    /// All measurements since `since_ms`, oldest first.
    pub fn latency_since(&self, since_ms: u64) -> Result<Vec<Sample>> {
        let db = self.db();
        let mut stmt =
            db.prepare("SELECT turn_id, at_ms, stage, ms FROM latency WHERE at_ms >= ?1 ORDER BY at_ms")?;
        let rows = stmt.query_map(params![since_ms as i64], |r| {
            Ok(Sample {
                turn: r.get::<_, Option<i64>>(0)?.map(|t| t as u64),
                at: r.get::<_, i64>(1)? as u64,
                stage: r.get(2)?,
                ms: r.get::<_, f64>(3)? as f32,
            })
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn audio(&self, id: u64, reply: bool) -> Option<Vec<u8>> {
        let col = if reply { "reply_wav" } else { "heard_wav" };
        self.db()
            .query_row(&format!("SELECT {col} FROM turns WHERE id = ?1"), params![id as i64], |r| r.get(0))
            .optional()
            .ok()
            .flatten()
            .flatten()
    }

    /// Newest first: the `limit` turns before `before` (or the latest ones).
    pub fn recent(&self, before: Option<u64>, limit: usize) -> Result<Vec<Turn>> {
        let db = self.db();
        let mut stmt = db.prepare(
            "SELECT id, at_ms, trigger, heard, heard_engine, heard_alt, alt_engine, intent, actions,
                    said, outcome, duration_ms, photo, heard_wav IS NOT NULL, reply_wav IS NOT NULL
             FROM turns WHERE id < ?1 ORDER BY id DESC LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![before.map_or(i64::MAX, |b| b as i64), limit as i64], |r| {
            let actions: String = r.get(8)?;
            Ok(Turn {
                id: r.get::<_, i64>(0)? as u64,
                at: r.get::<_, i64>(1)? as u64,
                trigger: r.get(2)?,
                heard: r.get(3)?,
                heard_engine: r.get(4)?,
                heard_alt: r.get(5)?,
                alt_engine: r.get(6)?,
                intent: r.get(7)?,
                actions: serde_json::from_str(&actions).unwrap_or_default(),
                said: r.get(9)?,
                outcome: r.get(10)?,
                duration_ms: r.get::<_, Option<i64>>(11)?.map(|d| d as u64),
                photo: r.get(12)?,
                heard_audio: r.get(13)?,
                reply_audio: r.get(14)?,
                latency: Default::default(),
                started: Instant::now(),
            })
        })?;
        let mut turns: Vec<Turn> = rows.collect::<Result<_, _>>()?;
        if let (Some(first), Some(last)) = (turns.last(), turns.first()) {
            let mut stmt = db.prepare(
                "SELECT turn_id, stage, ms FROM latency WHERE turn_id BETWEEN ?1 AND ?2",
            )?;
            let rows = stmt.query_map(params![first.id as i64, last.id as i64], |r| {
                Ok((r.get::<_, i64>(0)? as u64, r.get::<_, String>(1)?, r.get::<_, f64>(2)? as f32))
            })?;
            for row in rows {
                let (id, stage, ms) = row?;
                if let Some(t) = turns.iter_mut().find(|t| t.id == id) {
                    t.latency.insert(stage, ms);
                }
            }
        }
        Ok(turns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn version(c: &Connection) -> usize {
        c.pragma_query_value(None, "user_version", |r| r.get::<_, i64>(0)).unwrap() as usize
    }

    #[test]
    fn migrations() {
        let mut c = Connection::open_in_memory().unwrap();
        migrate(&mut c, MIGRATIONS).unwrap();
        assert_eq!(version(&c), MIGRATIONS.len());
        // idempotent
        migrate(&mut c, MIGRATIONS).unwrap();

        // a database from before migrations (tables, user_version 0) is adopted
        let mut old = Connection::open_in_memory().unwrap();
        old.execute_batch(MIGRATIONS[0]).unwrap();
        old.execute("INSERT INTO turns (at_ms, trigger) VALUES (1, 'wake word')", []).unwrap();
        migrate(&mut old, MIGRATIONS).unwrap();
        assert_eq!(version(&old), MIGRATIONS.len());
        assert_eq!(old.query_row("SELECT COUNT(*) FROM turns", [], |r| r.get::<_, i64>(0)).unwrap(), 1);

        // a new migration adds a column; a failing one rolls back and keeps the version
        let mut more = MIGRATIONS.to_vec();
        more.push("ALTER TABLE turns ADD COLUMN lang TEXT;");
        migrate(&mut old, &more).unwrap();
        old.execute("UPDATE turns SET lang = 'de'", []).unwrap();
        more.push("CREATE TABLE x (a); SELECT nope FROM nowhere;");
        assert!(migrate(&mut old, &more).is_err());
        assert_eq!(version(&old), MIGRATIONS.len() + 1);
        assert!(old.prepare("SELECT * FROM x").is_err());

        // newer database than the program
        assert!(migrate(&mut old, MIGRATIONS).is_err());
    }

    #[test]
    fn roundtrip_and_audio_pruning() {
        let dir = std::env::temp_dir().join(format!("uno-history-{}", std::process::id()));
        let path = dir.join("h.db");
        let h = History::open(path.to_str().unwrap(), 2).unwrap();
        let mut ids = Vec::new();
        for i in 0..3 {
            let id = h.insert(1000 + i, "wake word").unwrap();
            let mut t = h.recent(None, 1).unwrap().remove(0);
            t.heard = Some(format!("what time is it {i}"));
            t.actions = vec!["did something".into()];
            h.save(&t).unwrap();
            h.set_audio(id, false, b"RIFFfake").unwrap();
            ids.push(id);
        }
        let all = h.recent(None, 10).unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].heard.as_deref(), Some("what time is it 2"));
        assert_eq!(all[0].actions, vec!["did something".to_string()]);
        // keep_audio = 2: the oldest turn lost its audio
        assert!(h.audio(ids[0], false).is_none());
        assert_eq!(h.audio(ids[2], false).as_deref(), Some(&b"RIFFfake"[..]));
        assert_eq!(h.recent(Some(ids[2]), 10).unwrap().len(), 2);
        h.add_latency(&Sample { turn: Some(ids[1]), at: 1001, stage: "stt".into(), ms: 640.0 }).unwrap();
        h.add_latency(&Sample { turn: None, at: 1500, stage: "wake".into(), ms: 80.0 }).unwrap();
        assert_eq!(h.recent(None, 10).unwrap()[1].latency.get("stt"), Some(&640.0));
        assert_eq!(h.latency_since(1100).unwrap().len(), 1);
        std::fs::remove_dir_all(dir).ok();
    }
}
