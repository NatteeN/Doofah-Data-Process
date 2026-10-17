const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
  "Content-Type": "application/json; charset=utf-8",
};

function jsonResponse(body, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders, ...extraHeaders },
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders });
    }

    if (request.method !== "GET" && request.method !== "HEAD") {
      return jsonResponse({ status: "method_not_allowed" }, 405, {
        "Allow": "GET, HEAD, OPTIONS",
      });
    }

    if (url.pathname === "/health") {
      return jsonResponse({ status: "ok", service: "doofah-nowcast-api" }, 200, {
        "Cache-Control": "no-store",
      });
    }

    if (url.pathname !== "/nowcast/latest") {
      return jsonResponse({ status: "not_found", path: url.pathname }, 404, {
        "Cache-Control": "no-store",
      });
    }

    const value = await env.NOWCAST_KV.get("nowcast/latest.json");
    if (value === null) {
      return jsonResponse(
        {
          status: "not_ready",
          message: "Nowcast has not been published to KV yet.",
        },
        503,
        { "Cache-Control": "no-store" },
      );
    }

    // The producer is expected to publish valid JSON. Do not wrap the payload:
    // the frontend receives the original JSON document.
    return new Response(request.method === "HEAD" ? null : value, {
      status: 200,
      headers: {
        ...corsHeaders,
        "Cache-Control": "public, max-age=30",
      },
    });
  },
};
