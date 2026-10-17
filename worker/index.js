
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
    const pathname = url.pathname.replace(/\/+$/, "") || "/";

    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: corsHeaders,
      });
    }

    if (!["GET", "HEAD"].includes(request.method)) {
      return jsonResponse(
        { status: "method_not_allowed" },
        405,
        { Allow: "GET, HEAD, OPTIONS" }
      );
    }

    // Health check
    if (pathname === "/health") {
      return jsonResponse(
        { status: "ok", service: "doofah-nowcast-api" },
        200,
        { "Cache-Control": "no-store" }
      );
    }

    // Metadata endpoint เดิม
    if (pathname === "/nowcast/latest") {
      const value = await env.NOWCAST_KV.get("nowcast/latest.json");

      if (value === null) {
        return jsonResponse(
          { status: "not_ready", message: "Nowcast metadata is not available." },
          503,
          { "Cache-Control": "no-store" }
        );
      }

      return new Response(request.method === "HEAD" ? null : value, {
        status: 200,
        headers: {
          ...corsHeaders,
          "Cache-Control": "public, max-age=30",
        },
      });
    }

    // อ่านไฟล์ที่อยู่ภายใน Bundle
    if (!pathname.startsWith("/nowcast/")) {
      return jsonResponse({ status: "not_found", path: pathname }, 404);
    }

    const filename = pathname.slice("/nowcast/".length);

    // รับเฉพาะชื่อไฟล์ ไม่รับ path ซ้อน
    if (
      !filename ||
      filename.includes("/") ||
      filename.includes("..") ||
      filename.includes("\\")
    ) {
      return jsonResponse({ status: "not_found", path: pathname }, 404);
    }

    let bundleKey;

    if (
      filename.endsWith(".geojson") ||
      filename === "nowcast_metadata.json"
    ) {
      bundleKey = "nowcast_geojson_bundle";
    } else if (filename.endsWith(".json")) {
      bundleKey = "nowcast_grid_bundle";
    } else {
      return jsonResponse({ status: "not_found", path: pathname }, 404);
    }

    try {
      const bundle = await env.NOWCAST_KV.get(bundleKey, "json");

      if (!bundle || !Object.hasOwn(bundle, filename)) {
        return jsonResponse(
          { status: "not_found", filename, bundle: bundleKey },
          404
        );
      }

      return new Response(
        request.method === "HEAD"
          ? null
          : JSON.stringify(bundle[filename]),
        {
          status: 200,
          headers: {
            ...corsHeaders,
            "Cache-Control": "public, max-age=60",
          },
        }
      );
    } catch (error) {
      console.error("Nowcast bundle read failed:", error);

      return jsonResponse(
        { status: "internal_error", message: "Unable to read Nowcast bundle." },
        500,
        { "Cache-Control": "no-store" }
      );
    }
  },
};
