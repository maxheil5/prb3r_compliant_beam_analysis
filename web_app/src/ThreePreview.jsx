import { useEffect, useRef } from "react";
import * as THREE from "three";

import { GRIPPER_HALF_SPACING, beamCurve, mirroredPath } from "./geometry.js";

export default function ThreePreview({ caseRow }) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) {
      return undefined;
    }

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf7faf9);
    const camera = new THREE.PerspectiveCamera(32, 1.6, 0.1, 20);
    camera.position.set(1.35, -3.1, 1.65);
    camera.lookAt(0.46, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, Math.max(230, mount.clientWidth * 0.56));
    mount.appendChild(renderer.domElement);

    scene.add(new THREE.HemisphereLight(0xffffff, 0xb6c3c0, 2.4));
    const key = new THREE.DirectionalLight(0xffffff, 2.0);
    key.position.set(2.5, -3, 3);
    scene.add(key);

    const group = new THREE.Group();
    scene.add(group);
    sceneRef.current = { scene, camera, renderer, group };
    renderer.render(scene, camera);

    const resizeObserver = new ResizeObserver(() => {
      const width = mount.clientWidth;
      const height = Math.max(230, width * 0.56);
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.render(scene, camera);
    });
    resizeObserver.observe(mount);

    return () => {
      resizeObserver.disconnect();
      sceneRef.current = null;
      mount.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, []);

  useEffect(() => {
    const current = sceneRef.current;
    if (!current || !caseRow) {
      return;
    }
    drawGripper(current.group, caseRow);
    current.renderer.render(current.scene, current.camera);
  }, [caseRow]);

  return <div className="three-preview" ref={mountRef} />;
}

function drawGripper(group, caseRow) {
  group.clear();
  const material = new THREE.MeshStandardMaterial({ color: 0x58a99c, roughness: 0.48, metalness: 0.08 });
  const fixtureMaterial = new THREE.MeshStandardMaterial({ color: 0x46515b, roughness: 0.6 });

  const fixtureHeight = GRIPPER_HALF_SPACING * 2.0 + 0.42;
  const fixture = new THREE.Mesh(new THREE.BoxGeometry(0.1, fixtureHeight, 0.2), fixtureMaterial);
  fixture.position.set(0, 0, 0);
  group.add(fixture);

  const baseCurve = beamCurve(caseRow.Qx_ref_display, caseRow.Qy_ref_display, 34);
  addArm(group, mirroredPath(baseCurve, "upper"), material);
  addArm(group, mirroredPath(baseCurve, "lower"), material);
}

function addArm(group, points, material) {
  for (let index = 0; index < points.length - 1; index += 1) {
    const a = points[index];
    const b = points[index + 1];
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const length = Math.hypot(dx, dy);
    const segment = new THREE.Mesh(new THREE.BoxGeometry(length, 0.028, 0.12), material);
    segment.position.set((a.x + b.x) / 2, (a.y + b.y) / 2, 0);
    segment.rotation.z = Math.atan2(dy, dx);
    group.add(segment);
  }
}
