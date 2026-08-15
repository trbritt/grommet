//! Whole-stack deterministic network simulation for both public protocols.
//!
//! Turmoil supplies the sockets, so HTTP and gRPC bytes cross a simulated
//! network that can be partitioned and repaired on demand. Nothing below the
//! socket is mocked: this is the real router, the real shard reactor and the
//! real processor.
#![cfg(sim)]

use accounts::frontdoor::{AccountRouter, Frontdoor};
use accounts::grpc::account_processor_client::AccountProcessorClient;
use accounts::grpc::{SubmitRequest, submit_request};
use accounts::processor::AccountCall;
use accounts::sim::{FaultPoint, Plan, SimWorld};
use bytes::Bytes;
use grommet::metrics::ShardStats;
use grommet::{ManualClock, ShardConfig, shard};
use http_body_util::{BodyExt, Full};
use hyper_util::rt::TokioIo;
use std::sync::Arc;
use std::time::Duration;

type Error = Box<dyn std::error::Error>;

/// One host running a shard and one protocol server.
fn spawn_engine(sim: &mut turmoil::Sim<'_>, plan: Plan, grpc: bool) {
    sim.host("engine", move || {
        let plan = plan.clone();
        async move {
            let world = SimWorld::new(plan);
            let clock = ManualClock::new();
            let (tx, rx) = tokio::sync::mpsc::channel::<grommet::Envelope<AccountCall>>(32);
            let router = Arc::new(AccountRouter::<ManualClock>::new(vec![tx], clock.clone()));
            let mut cfg = ShardConfig::new([64, 8]);
            cfg.tick = Duration::from_millis(50);
            let engine =
                shard::run(rx, world.processor(), clock, Arc::new(ShardStats::default()), cfg);
            let frontdoor = Frontdoor::new(router);
            if grpc {
                let server = accounts::frontdoor::serve_grpc(frontdoor, 9001);
                tokio::select! {
                    _ = engine => Ok(()),
                    result = server => result.map_err(|error| {
                        Box::<dyn std::error::Error>::from(std::io::Error::other(error.to_string()))
                    }),
                }
            } else {
                let server = accounts::frontdoor::serve_http(frontdoor, 9000);
                tokio::select! {
                    _ = engine => Ok(()),
                    result = server => result.map_err(Into::into),
                }
            }
        }
    });
}

async fn http_submit(req_id: &str) -> Result<(hyper::StatusCode, String), Error> {
    let body = serde_json::json!({
        "req_id": req_id,
        "account": 42,
        "op": "credit",
        "amount": 100
    })
    .to_string();
    let stream = accounts::net::TcpStream::connect(("engine", 9000)).await?;
    let (mut sender, connection) =
        hyper::client::conn::http1::handshake(TokioIo::new(stream)).await?;
    tokio::spawn(async move {
        let _ = connection.await;
    });
    let request = hyper::Request::builder()
        .method("POST")
        .uri("/v1/submit")
        .header("host", "engine")
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)))?;
    let response = sender.send_request(request).await?;
    let status = response.status();
    let bytes = response.into_body().collect().await?.to_bytes();
    Ok((status, String::from_utf8_lossy(&bytes).into_owned()))
}

#[test]
fn a_partition_and_an_in_doubt_retry_cross_the_whole_stack() -> turmoil::Result {
    let mut sim = turmoil::Builder::new().simulation_duration(Duration::from_secs(30)).build();
    spawn_engine(&mut sim, Plan::ordered([FaultPoint::CommitAfterApply]), false);

    sim.client("client", async {
        let req_id = accounts::domain::RequestId::from(0x1234u128).to_string();

        // The commit applies durably but its acknowledgement is lost.
        let (status, _) = http_submit(&req_id).await?;
        assert_eq!(status, hyper::StatusCode::SERVICE_UNAVAILABLE);

        // The client cannot even reach the service to find out.
        turmoil::partition("client", "engine");
        assert!(http_submit(&req_id).await.is_err());
        turmoil::repair("client", "engine");

        // Retrying under the same id discovers the truth: it had applied.
        let (status, body) = http_submit(&req_id).await?;
        assert_eq!(status, hyper::StatusCode::OK);
        assert!(body.contains("\"balance\":100"), "{body}");
        assert!(body.contains("\"duplicate\":true"), "{body}");
        Ok(())
    });
    sim.run()
}

#[test]
fn grpc_crosses_the_simulated_network_and_recognises_a_replay() -> turmoil::Result {
    let mut sim = turmoil::Builder::new().simulation_duration(Duration::from_secs(30)).build();
    spawn_engine(&mut sim, Plan::off(), true);

    sim.client("client", async {
        let channel = tonic::transport::Endpoint::from_static("http://engine:9001")
            .connect_with_connector(accounts::net::connector::SimConnector)
            .await?;
        let mut client = AccountProcessorClient::new(channel);
        let req_id = accounts::domain::RequestId::from(0x5678u128).to_string();
        let request = SubmitRequest {
            req_id: req_id.clone(),
            account: 9,
            kind: submit_request::Kind::Credit as i32,
            amount: 7,
            scenarios: 0,
        };

        let first = client.submit(request.clone()).await?.into_inner();
        assert_eq!(first.balance, 7);
        assert!(!first.duplicate);

        let replay = client.submit(request).await?.into_inner();
        assert_eq!(replay.balance, 7);
        assert!(replay.duplicate, "the same id must not credit twice");

        let revalued = client
            .submit(SubmitRequest {
                req_id: accounts::domain::RequestId::from(0x5679u128).to_string(),
                account: 9,
                kind: submit_request::Kind::Revalue as i32,
                amount: 0,
                scenarios: 0,
            })
            .await?
            .into_inner();
        assert_eq!(revalued.balance, 7);
        Ok(())
    });
    sim.run()
}
