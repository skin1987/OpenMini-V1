use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct AdminConfig {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub jwt: JwtConfig,
    pub upstream: UpstreamConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
}

fn default_host() -> String {
    "0.0.0.0".into()
}
fn default_port() -> u16 {
    9000
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatabaseConfig {
    #[serde(default = "default_db_url")]
    pub url: String,
}

fn default_db_url() -> String {
    "sqlite:data/admin.db?mode=rwc".into()
}

#[derive(Debug, Clone, Deserialize)]
pub struct JwtConfig {
    #[serde(default = "default_jwt_secret")]
    pub secret: String,
    #[serde(default = "default_jwt_exp")]
    pub expiration_hours: u64,
}

fn default_jwt_secret() -> String {
    "openmini-admin-jwt-secret-key-change-in-production".into()
}
fn default_jwt_exp() -> u64 {
    24
}

#[derive(Debug, Clone, Deserialize)]
pub struct UpstreamConfig {
    #[serde(default = "default_upstream_url")]
    pub base_url: String,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

fn default_upstream_url() -> String {
    "http://localhost:8080/api/v1".into()
}
fn default_timeout() -> u64 {
    10
}

impl Default for AdminConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: default_host(),
                port: default_port(),
            },
            database: DatabaseConfig {
                url: default_db_url(),
            },
            jwt: JwtConfig {
                secret: default_jwt_secret(),
                expiration_hours: default_jwt_exp(),
            },
            upstream: UpstreamConfig {
                base_url: default_upstream_url(),
                timeout_secs: default_timeout(),
            },
        }
    }
}

pub fn load_config() -> Option<AdminConfig> {
    let path = std::path::Path::new("config/admin.toml");
    if path.exists() {
        let content = std::fs::read_to_string(path).ok()?;
        toml::from_str(&content).ok()
    } else {
        Some(AdminConfig::default())
    }
}
