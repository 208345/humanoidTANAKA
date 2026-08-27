using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class HumanoidAgent : Agent
{
    [Header("Robot Parts (Drag & Drop here)")]
    [Tooltip("ルート（胴体）のArticulationBodyをアタッチしてください")]
    public ArticulationBody torso;

    [Tooltip("6つの関節（joint_1 〜 joint_6）のArticulationBodyを順にアタッチしてください")]
    public ArticulationBody[] joints = new ArticulationBody[6];
    
    [Header("Control Settings")]
    [Tooltip("Pythonからのアクション(-1.0〜1.0)に乗算する最大角度（度）")]
    public float maxAngleDeg = 90f;

    // 前回のステップでの位置・角度などを記憶しておく
    private Vector3 initialPosition;
    private Quaternion initialRotation;

    public override void Initialize()
    {
        if (torso != null)
        {
            initialPosition = torso.transform.position;
            initialRotation = torso.transform.rotation;
        }
        else
        {
            Debug.LogError("Torso ArticulationBody が設定されていません！");
        }
    }

    public override void OnEpisodeBegin()
    {
        if (torso == null) return;

        // 胴体を初期位置に戻す
        torso.TeleportRoot(initialPosition, initialRotation);
        torso.velocity = Vector3.zero;
        torso.angularVelocity = Vector3.zero;

        // 関節を初期位置に戻す
        foreach (var joint in joints)
        {
            if (joint != null)
            {
                var drive = joint.xDrive;
                drive.target = 0f;
                joint.xDrive = drive;
                joint.jointPosition = new ArticulationReducedSpace(0f);
                joint.jointVelocity = new ArticulationReducedSpace(0f);
            }
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // 1. 関節の角度 (6)
        foreach (var joint in joints)
        {
            if (joint != null) sensor.AddObservation(joint.jointPosition[0]);
            else sensor.AddObservation(0f);
        }

        // 2. 関節の角速度 (6)
        foreach (var joint in joints)
        {
            if (joint != null) sensor.AddObservation(joint.jointVelocity[0]);
            else sensor.AddObservation(0f);
        }

        if (torso != null)
        {
            // 3. 胴体のクォータニオン (4)
            // Python側では (w, x, y, z) の順を期待している
            Quaternion rot = torso.transform.rotation;
            sensor.AddObservation(rot.w);
            sensor.AddObservation(rot.x);
            sensor.AddObservation(rot.y);
            sensor.AddObservation(rot.z);

            // 4. 胴体の角速度 (3)
            Vector3 angVel = torso.angularVelocity;
            sensor.AddObservation(angVel.x);
            sensor.AddObservation(angVel.y);
            sensor.AddObservation(angVel.z);
        }
        else
        {
            for(int i=0; i<7; i++) sensor.AddObservation(0f);
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var continuousActions = actionBuffers.ContinuousActions;
        for (int i = 0; i < 6; i++)
        {
            if (joints[i] != null && i < continuousActions.Length)
            {
                var drive = joints[i].xDrive;
                drive.target = continuousActions[i] * maxAngleDeg;
                joints[i].xDrive = drive;
            }
        }
    }
    
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        for (int i = 0; i < 6; i++) continuousActionsOut[i] = 0f;
    }
}
