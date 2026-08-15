//! The only socket types named by the service.
//!
//! Under the `sim` feature they are Turmoil sockets, so HTTP and gRPC bytes use
//! the simulated network. Shipping builds use Tokio. A release tripwire rejects
//! an accidentally unified simulation feature.

#[cfg(not(feature = "sim"))]
pub use tokio::net::{TcpListener, TcpStream};

#[cfg(feature = "sim")]
pub use turmoil::net::{TcpListener, TcpStream};

use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

/// Supplies Tonic's connection-metadata marker for both Tokio and Turmoil
/// streams while delegating every byte unchanged.
pub struct ServerIo<T>(pub T);

impl<T> tonic::transport::server::Connected for ServerIo<T> {
    type ConnectInfo = ();

    fn connect_info(&self) -> Self::ConnectInfo {}
}

impl<T: AsyncRead + Unpin> AsyncRead for ServerIo<T> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.0).poll_read(cx, buffer)
    }
}

impl<T: AsyncWrite + Unpin> AsyncWrite for ServerIo<T> {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &[u8],
    ) -> Poll<Result<usize, std::io::Error>> {
        Pin::new(&mut self.0).poll_write(cx, buffer)
    }

    fn poll_flush(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<(), std::io::Error>> {
        Pin::new(&mut self.0).poll_flush(cx)
    }

    fn poll_shutdown(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<(), std::io::Error>> {
        Pin::new(&mut self.0).poll_shutdown(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[tokio::test]
    async fn server_io_forwards_read_write_flush_and_shutdown() {
        let (stream, mut peer) = tokio::io::duplex(16);
        let mut io = ServerIo(stream);
        io.write_all(b"ok").await.unwrap();
        io.flush().await.unwrap();
        let mut bytes = [0; 2];
        peer.read_exact(&mut bytes).await.unwrap();
        assert_eq!(&bytes, b"ok");
        io.shutdown().await.unwrap();
    }
}

#[cfg(all(feature = "sim", not(debug_assertions), not(sim)))]
compile_error!("feature `sim` leaked into an optimized non-simulation build");

#[cfg(feature = "sim")]
pub mod connector {
    use super::TcpStream;
    use hyper_util::rt::TokioIo;
    use std::future::Future;
    use std::pin::Pin;
    use tower::Service;

    #[derive(Clone, Default)]
    pub struct SimConnector;

    impl Service<hyper::Uri> for SimConnector {
        type Response = TokioIo<TcpStream>;
        type Error = std::io::Error;
        type Future =
            Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send + 'static>>;

        fn poll_ready(
            &mut self,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            std::task::Poll::Ready(Ok(()))
        }

        fn call(&mut self, uri: hyper::Uri) -> Self::Future {
            Box::pin(async move {
                let host = uri.host().unwrap_or("localhost").to_owned();
                let port = uri.port_u16().unwrap_or(9000);
                TcpStream::connect((host, port)).await.map(TokioIo::new)
            })
        }
    }
}
