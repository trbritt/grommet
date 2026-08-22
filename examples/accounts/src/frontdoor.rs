//! HTTP and gRPC codecs over one account router.

use crate::domain::{Op, Reply, RequestId, RevalueParams};
use crate::grpc::account_processor_server::{AccountProcessor, AccountProcessorServer};
use crate::grpc::{SubmitRequest as GrpcRequest, SubmitResponse as GrpcResponse};
use crate::processor::{AccountCall, Request};
use axum::{Json, Router as AxumRouter, extract::State, http::StatusCode, routing::post};
use grommet::{CallError, Clock, Router, SubmitError};
use hyper_util::rt::{TokioExecutor, TokioIo};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The router shape this service uses: two classes, replies attached. The
/// class count is the runtime's default, so it does not need naming.
pub type AccountRouter<C> = Router<AccountCall, C>;

pub struct Frontdoor<C: Clock> {
    router: Arc<AccountRouter<C>>,
}

// A manual impl: `Arc` is cloneable whatever `C` is, but derive would demand
// `C: Clone` on the struct itself.
impl<C: Clock> Clone for Frontdoor<C> {
    fn clone(&self) -> Self {
        Self { router: self.router.clone() }
    }
}

impl<C: Clock> Frontdoor<C> {
    pub fn new(router: Arc<AccountRouter<C>>) -> Self {
        Self { router }
    }

    /// Submit one request and translate whatever comes back into a reply.
    ///
    /// Rejection is a first-class answer here: a shed request is reported as
    /// such rather than being retried invisibly or dropped.
    async fn submit(&self, request: Request) -> Reply {
        match self.router.call(request).await {
            Ok(reply) => reply,
            Err(CallError::Rejected(SubmitError::Full(_))) => {
                Reply::Err("overloaded; retry with the same request id".to_owned())
            }
            Err(CallError::Rejected(SubmitError::ShardDown(_))) => {
                Reply::Err("shard unavailable".to_owned())
            }
            Err(CallError::Rejected(SubmitError::InvalidClass(_))) => {
                Reply::Err("unroutable operation".to_owned())
            }
            Err(CallError::Cancelled) => {
                Reply::Err("request was dropped without a response".to_owned())
            }
            // `CallError` and `SubmitError` are `#[non_exhaustive]`, so a
            // newer grommet can report a rejection this build has never heard
            // of. Refusing the request is the only safe answer to that: the
            // work did not run, and guessing which of the arms above it
            // resembles would be inventing a reason.
            Err(other) => Reply::Err(other.to_string()),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct SubmitBody {
    pub req_id: String,
    pub account: u64,
    pub op: String,
    #[serde(default)]
    pub amount: i64,
    #[serde(default)]
    pub scenarios: u32,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SubmitOut {
    pub balance: i64,
    pub duplicate: bool,
}

fn parse_request(
    req_id: &str,
    account: u64,
    op: &str,
    amount: i64,
    scenarios: u32,
) -> Result<Request, String> {
    let req_id = req_id.parse::<RequestId>().map_err(|_| "invalid ULID".to_owned())?;
    let op = match op {
        "debit" => Op::Debit(amount),
        "credit" => Op::Credit(amount),
        "balance" => Op::Balance,
        "revalue" => Op::Revalue(RevalueParams { scenarios }),
        _ => return Err("unknown operation".to_owned()),
    };
    Ok(Request { req_id, account, op })
}

fn map_reply(reply: Reply) -> Result<SubmitOut, String> {
    match reply {
        Reply::Ok(balance) => Ok(SubmitOut { balance, duplicate: false }),
        Reply::Duplicate(balance) => Ok(SubmitOut { balance, duplicate: true }),
        Reply::Err(error) => Err(error),
    }
}

pub fn http_router<C: Clock>(frontdoor: Frontdoor<C>) -> AxumRouter {
    AxumRouter::new()
        .route("/v1/submit", post(http_submit::<C>))
        .route("/healthz", axum::routing::get(|| async { "ok" }))
        .with_state(frontdoor)
}

async fn http_submit<C: Clock>(
    State(frontdoor): State<Frontdoor<C>>,
    Json(body): Json<SubmitBody>,
) -> Result<Json<SubmitOut>, (StatusCode, String)> {
    let request = parse_request(&body.req_id, body.account, &body.op, body.amount, body.scenarios)
        .map_err(|error| (StatusCode::BAD_REQUEST, error))?;
    let reply = frontdoor.submit(request).await;
    map_reply(reply).map(Json).map_err(|error| (StatusCode::SERVICE_UNAVAILABLE, error))
}

#[tonic::async_trait]
impl<C: Clock> AccountProcessor for Frontdoor<C> {
    async fn submit(
        &self,
        request: tonic::Request<GrpcRequest>,
    ) -> Result<tonic::Response<GrpcResponse>, tonic::Status> {
        let request = request.into_inner();
        let op_name = match request.kind() {
            crate::grpc::submit_request::Kind::Debit => "debit",
            crate::grpc::submit_request::Kind::Credit => "credit",
            crate::grpc::submit_request::Kind::Balance => "balance",
            crate::grpc::submit_request::Kind::Revalue => "revalue",
            crate::grpc::submit_request::Kind::Unspecified => {
                return Err(tonic::Status::invalid_argument("operation is required"));
            }
        };
        let parsed = parse_request(
            &request.req_id,
            request.account,
            op_name,
            request.amount,
            request.scenarios,
        )
        .map_err(tonic::Status::invalid_argument)?;
        match map_reply(Frontdoor::submit(self, parsed).await) {
            Ok(reply) => Ok(tonic::Response::new(GrpcResponse {
                balance: reply.balance,
                duplicate: reply.duplicate,
                error: String::new(),
            })),
            Err(error) => Err(tonic::Status::unavailable(error)),
        }
    }
}

#[cfg_attr(coverage, coverage(off))]
pub async fn serve_http<C: Clock>(frontdoor: Frontdoor<C>, port: u16) -> std::io::Result<()> {
    let listener = crate::net::TcpListener::bind(("0.0.0.0", port)).await?;
    let app = http_router(frontdoor);
    loop {
        let (stream, _) = listener.accept().await?;
        let service = app.clone();
        tokio::spawn(async move {
            let io = TokioIo::new(stream);
            let _ = hyper_util::server::conn::auto::Builder::new(TokioExecutor::new())
                .serve_connection(io, hyper_util::service::TowerToHyperService::new(service))
                .await;
        });
    }
}

#[cfg_attr(coverage, coverage(off))]
pub async fn serve_grpc<C: Clock>(
    frontdoor: Frontdoor<C>,
    port: u16,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let listener = crate::net::TcpListener::bind(("0.0.0.0", port)).await?;
    let incoming = futures::stream::unfold(listener, |listener| async move {
        let next = listener.accept().await.map(|(stream, _)| crate::net::ServerIo(stream));
        Some((next, listener))
    });
    tonic::transport::Server::builder()
        .add_service(AccountProcessorServer::new(frontdoor))
        .serve_with_incoming(incoming)
        .await
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_protocol_parser_requires_a_ulid_and_a_known_operation() {
        let id = RequestId::from(1u128).to_string();
        assert!(matches!(
            parse_request(&id, 1, "credit", 5, 0),
            Ok(Request { op: Op::Credit(5), account: 1, .. })
        ));
        assert!(matches!(
            parse_request(&id, 1, "debit", 5, 0),
            Ok(Request { op: Op::Debit(5), .. })
        ));
        assert!(matches!(
            parse_request(&id, 1, "balance", 0, 0),
            Ok(Request { op: Op::Balance, .. })
        ));
        assert!(matches!(
            parse_request(&id, 1, "revalue", 0, 3),
            Ok(Request { op: Op::Revalue(RevalueParams { scenarios: 3 }), .. })
        ));
        assert_eq!(
            parse_request("not-a-ulid", 1, "credit", 5, 0).unwrap_err(),
            "invalid ULID".to_owned()
        );
        assert_eq!(parse_request(&id, 1, "wat", 0, 0).unwrap_err(), "unknown operation".to_owned());
    }

    #[test]
    fn replies_map_onto_the_wire_shape() {
        assert_eq!(map_reply(Reply::Ok(5)), Ok(SubmitOut { balance: 5, duplicate: false }));
        assert_eq!(map_reply(Reply::Duplicate(5)), Ok(SubmitOut { balance: 5, duplicate: true }));
        assert_eq!(map_reply(Reply::Err("nope".to_owned())), Err("nope".to_owned()));
    }
}
